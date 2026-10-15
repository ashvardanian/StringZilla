# Regex: Multi-Pattern Regular Expressions Over Two Automata

This directory holds the kernels behind `sz_regex_engine_init`, `sz_regex_counts`, `sz_regex_find`, `sz_regex_replace` and `sz_regex_extract`.
One engine owns a compiled vocabulary of RE2-syntax patterns, and every haystack then streams through it whatever the pattern count.
Matches are leftmost-first across the union of patterns, ties going to the lower pattern index, which is what Rust's `regex`, RE2, Python's `re` and Arrow report.
On valid UTF-8 the serial backend reproduces `regex-automata`'s `find_iter` exactly, and its tests check that against the crate itself.

## The Automata

A vocabulary compiles into two DFAs over one byte-class map.
A class groups bytes that no pattern tells apart: no range of the NFA separates them, and neither a line anchor nor a word boundary does.
Both automata are dense tables of one row per state and one column per class, plus a last column taken at a haystack's edge, that block repeating once per set of lookarounds that may hold, as [Lookarounds](#lookarounds) describes.
A state id is its row's offset, so a step is one load with no multiply; the dead state is zero and accepting states come last, so acceptance is one compare.

The __start-finding__ automaton is the patterns reversed and unanchored, keeping every match.
Walked right to left, it accepts after the byte at `i` exactly when some match starts at `i + 1`, so one pass marks every position a match can start at.

The __end-finding__ automaton is the patterns anchored and leftmost-first.
Walked left to right from one start until it dies, its last accepting state names the match's end and the pattern that leftmost-first priority picks.

Both are built the way `regex-automata` builds its dense DFAs, so its semantics carry over rather than being re-derived.
The parser follows RE2's syntax and `regex-syntax`'s simplifications, and the Thompson NFA follows its compiler, which fixes how priority treats empty iterations: `x*` of an `x` that can match empty compiles as `(x+)?`.
A DFA state is an ordered set of NFA states; under leftmost-first, every state after the first match in that order is cut.
A match is reported one byte late, which lets look-ahead resolve: `$`, `\z` and the far side of `\b` are settled by the byte the next step consumes, and by the edge column at a haystack's end.
Look-behind is settled by the start state, one per kind of byte before the walk: none, a line feed, a word byte, or another.

## The Walks

A haystack splits into equal windows of at most 256 KB, eight to a batch, left to right.
The search marks a whole batch at once: eight start-finding walks step right to left in lockstep, one per window, each from the state at its window's right edge, so their dependent loads overlap.
A DFA chain gathers its start bits in a register, 64 steps at a time; position sets walk four at a time, as each step holds many more values.
A cased DFA walk finds those states almost for free, as the section on speculation below tells, so its windows can be as short as 128 bytes; uncased and position walks prime every window, and a haystack too short for a batch of their 4 KB windows is one window, walked alone.
Short windows only pay in lockstep, so a haystack under 32 KB whose sampled bytes would skip, as below, stays one window too.
Where literal prefixes replace the start pass, below, a haystack stays one window.

While the start pass idles, a reverse byte-set search can jump to the next byte that leaves it, as nothing between can start a match.
The DFA idles in its start states after a line feed, a word byte and another byte, as long as every byte outside the live set maps those three into one another; after a jump it lands on the start for the kind of the last byte it skipped, so vocabularies ending in `\b` skip too.
The position walk idles when only the loop is live, unless the loop alone can match empty.
A batch skips window by window instead of walking in lockstep when that is cheaper, judged first by 64 sampled bytes and then by what skipping cost: a DFA step on one chain costs about three lockstep bytes and a search about forty, while a position step costs about as much as its lockstep step and a search about four.
The samples let a DFA skip below one live byte in 46 on the serial tier and one in 20 where vector searches pass a block per compare.
A rare live byte can still hold the walk off its idle states for long, like the opening quote of `"[^"]*"` in a quoted CSV, so the first batch that samples well only tries one window before deciding.
A skipping DFA batch walks right to left as one walk, each window entered with the state the window right of it ended on.

The search then follows the chain the sequential search takes: the first marked start at or past the cursor, its end from the end-finding walk, and the cursor moved there.
The cost is one pass over the haystack for starts plus the matched bytes and their lookahead for ends, the same order as RE2's forward pass plus a reverse walk per match.

Where every match starts with a few literal bytes, as keywords do, the search skips the start pass and searches for those bytes instead.
The end-finding DFA's first steps, from every start state under every lookaround block, give the paths of classes the first bytes of a match follow, up to three bytes deep and eight paths wide, each path one bucket.
A search jumps to the next position whose bytes keep some bucket at every offset, testing both nibbles of every byte against per-offset bucket tables, and the end walk from there settles whether a match starts there, so a false candidate costs a walk that dies within a step or two.
The paths must pass few positions, judged with every byte at least a sixteenth as frequent as any, as text is: two literal bytes pass, a single byte or a wide class like `[0-9]` does not, and neither does a vocabulary that can match empty.
GPUs mark such candidates a slot per thread from the bucket tables, in place of the start-finding walk and its priming, and their end walks settle them as they settle any start.

An end walk that runs past its match, looking for a longer one, can cover the starts of the next matches: every `a` of a line without a later `z` walks to the line's end under `a[^\n]*z|a`.
So every 16 bytes a walk leaves its state at a checkpoint, and a later walk holding the same state there stops: it walks the same bytes from there on, so it takes that walk's match if that one ended past the checkpoint, or its own last match before it.
That keeps such vocabularies linear; it costs walks that never meet 2 to 5%.
At a checkpoint, a walk whose state stays put on the byte at hand also searches ahead for the next byte that moves it, as inside `"[^"]*"`, where the state loops on everything but the quote; a state qualifies when at most eight printable ASCII bytes move it.
The check stays at checkpoints, as searching after every escape loses where escapes are frequent.

## Tiers

Every CPU tier shares the serial walks and brings its byte-set searches: the reverse one for the start pass's skips, and the forward one for end walks.
NEON, AVX2 and AVX-512 search 16, 32 and 64 bytes per compare; the serial tier skips the start pass with a byte loop, which still pays for position walks, and does not search ahead in end walks, where a byte loop is no faster than stepping.
Each tier also brings its search for literal prefixes: the serial one reads a bucket table per offset, a byte at a time, and the vector ones test both nibbles of every byte at every offset with one table lookup each, a superset of the buckets' bytes that the end walk settles.
And each brings its fill of the lookaround table's one-byte lookarounds: AVX-512 narrows both 256-entry tables to bytes and looks up 64 positions per step with two byte permutations and a blend, NEON 16 with four four-register lookups, where every lookaround written is among the first eight; serial and AVX2, which has no 256-entry byte lookup cheaper than two scalar loads, read the tables a byte at a time.
The batch marking is compiled once outside every tier's target, as an AVX-512 target otherwise widens the eight chains into gathers, several times slower than the chains.

After a match the search resumes at its end, and an empty match there is refused in favour of the search one byte further, so matches share no bytes and none repeats.
An empty match is never reported before a UTF-8 continuation byte, so none splits a codepoint, while non-empty matches stand where they are found.
On valid UTF-8 that is `regex-automata`'s rule; on stray continuation bytes the crate instead drops every match ending before one whenever some pattern can match empty, which would cost the engine a flag for no valid haystack.

## The Bit-Parallel Start Pass

An unanchored automaton grows exponentially with bounded gaps: `\b(?:error|warning|fatal)\b.{0,40}\b(?:disk|memory|timeout)\b` needs 2.9 million start-finding states, while its anchored end-finding automaton needs 1,227.
So the start-finding subset construction stops at a budget of 16,384 states, about 50 ms of compilation, and the start pass runs on a position automaton of the same reversed NFA instead; end walks stay DFAs within a budget of their own, as the limits below describe.

A __position__ is one class of the reversed patterns, consuming a whole codepoint, so `.{0,40}` is forty positions rather than forty copies of the UTF-8 automaton for `.`.
Position zero is the unanchored loop, which consumes any unit and is always live, and position one is the match, entered by the step over the unit before a match start.
A step ORs the follow sets of the live positions, then ANDs the reach row of the unit's class.
Follows that move positions by one common distance, mostly to the next position of a sequence, are __shifts__; the rest are __groups__, sources that share the targets they enter.
Assertions become masks over 16 boundary kinds, the kinds of the units on either side, which a group checks before it enters its targets.

A malformed byte is a unit of its own that only the loop consumes, and no start is marked inside a codepoint, where the search never reports one.
Window edges move forward to the next unit boundary, so both start passes share one window layout.

Priming stays exact by __sandwich__: a step is monotone in its set, so a run from the loop alone and a run from every position a walk can hold bound the true set from both sides, and where they meet it is known.

## Priming and Speculation

A window that does not end its haystack needs the start-finding state at its right edge, which depends on every byte after it.
Uncased and position walks, and the GPUs, have each window __prime__ itself: every state of the automaton walks the bytes right of the edge together, deduplicated, until one state remains.
Whichever state the walk from the haystack's end would have held there, it is now that one, so the result is exact by construction.
A distance that ends uncollapsed is doubled, one that reaches the haystack's end starts from the true start instead, and every window edge passed once collapsed is recorded, so no window primes over the same bytes twice.
A primed state that dies is dropped on the spot, as the true walk never dies inside a haystack: the state after the edge column accepts with nothing left to run.

Most vocabularies collapse within a few bytes.
A pattern that carries state across lines or quotes, like `.*error.*` or `"[^"]*"`, collapses at the next line feed or quote, which can be tens of kilobytes away.
The alternative, a restarting forward search that guesses its entry state, never re-synchronizes on `"[^"]*"`, as the quote parity it would need is exactly what it lost.

A guess was measured on the GPU backends anyway, where priming takes launches of its own.
Each chunk walked its own bytes from the idle state after the byte at its edge and, beside it, the set priming walks, so once that set narrowed to one state the marks and the state handed left were exact, and only the bytes before that point waited for the neighbour's state.
Where nearly every pair of bytes collapses the start-finding states, as for keywords and everyday patterns, that gained a few percent: the priming it saves is a fraction of a pass, while the chunks it leaves unsettled, at least one per block, still take priming's launches.
Where no pair does, as for `"[^"]*"`, every chunk fell back to priming, and the guess cost a quarter more.

A cased DFA batch on CPUs primes only its last window, whose state at the right edge comes from priming or the haystack's end, and guesses the rest without launches to save.
Each guess walks from an idle state over the 16 bytes right of its window's edge, all windows in lockstep.
Once every window is walked, the guesses are settled right to left against the state the window right of each ended on, one step into the window, as a window's first step marks its right edge, which only the next window is searched for.
A wrong guess walks on beside the true state until the two meet, flipping the start bits where exactly one accepts; for `"[^"]*"` they meet at the next quote, the bytes priming would have walked, and a pair that never meets hands the true state on.
That lets lines of a few kilobytes, which used to be one window each, walk in eight-way lockstep.

## Rewriting and Groups

`sz_regex_replace` substitutes every match with its pattern's replacement, and `sz_regex_extract` locates the groups of each haystack's first match of one pattern, as Pandas `str.extract` reads them.
Extraction writes `find`'s rows: one for the whole match, `needle_index` zero, then one per group that took part, `needle_index` naming it, so a group left out differs from an empty one, and a haystack without a match gets no rows.
Plain and named groups capture, numbered from one per pattern by their opening parenthesis.
A replacement names them as `$N`, `${N}` or `${name}`, `$0` being the whole match and `$$` one dollar; a group that took no part or does not exist is empty, and a dollar starting no reference stands for itself, so `$1a` is group one and an `a`.

Neither automaton tracks groups, so the engine keeps the forward Thompson NFA, whose capture states are empty states carrying slots, with one entry state per pattern.
A match's groups come from a PikeVM entered at its pattern's entry and run over the match's own span only, which reproduces the thread the leftmost-first union picked, and only for matches whose replacement names a group or that extraction reads.

## Lookarounds

Lookaheads `(?=X)` and `(?!X)` and lookbehinds `(?<=X)` and `(?<!X)` are regular in effect, as RegElk and Mamouras & Chattopadhyay evaluate them: first find, per haystack position, which of them hold, then let the walks read that like `\b`.
Each body compiles into the forward NFA as a sub-vocabulary of its own, a lookbehind's as written and a lookahead's reversed, and each lookaround owns one look bit.
Before the walks, every haystack fills a table of those bits per position: a forward pass runs all lookbehind bodies as one unanchored automaton keeping every match and marks where they end, and a backward pass runs the reversed lookahead bodies alike and marks where they start, which is the start-finding construction again; a negated lookaround holds where its body marked nothing.
Bodies nest, so lookarounds are computed level by level, innermost first, each level's bodies reading the bits below them.
A body nesting no lookaround gets a DFA of its own, unanchored and keeping every match as the start-finding one, and fills its bits a byte per step; nested bodies and those past the start-finding budget step the PikeVM.
Over a cased haystack of a kilobyte or more, a body's DFA walks four stretches in lockstep, each guessing its entry from the 16 bytes before it and settled against the stretch before it as the start pass's windows are.
A cased body of one byte from a set, asserting nothing, needs no walk at all: the byte before a position decides a lookbehind and the byte at it a lookahead, so one table per side fills all of them in one pass, which also marks every negated lookaround a DFA walks as holding until its body flips it.
GPUs fill the same tables a slot per thread, and step a body's DFA over each chunk as a set of its states, which crosses chunk edges the way the position sets do; a chunk that every state leaves in one state hands that one on.
The PikeVM adds a position's bits to the assertions it crosses there, and the position automaton tests them beside the boundary kinds of its groups, the loop's follows across a lookaround joining the groups too.
Both DFAs read the bits as columns: a vocabulary of at most six lookarounds, nested ones included, widens every row to one block per set of them, and a step takes the block of the lookarounds holding at the boundary it crosses, the one before its byte left to right and the one after it right to left.
Such a step first closes its state's threads under the assertions the state knows and the block's lookarounds, then consumes the byte, so a lookaround ending a pattern resolves one byte late like `$`, and each block's edge column settles the haystack's edges.
Steps inside the folds of one codepoint take the first block, as no lookaround holds there.
Each lookaround halves both state budgets, so the tables keep their bytes, and a vocabulary of more walks positions and the PikeVM.
A body holds no capturing group, and a vocabulary at most 26 lookarounds, the look bits left past the assertions.

## Case Folding

An uncased engine folds its patterns at compile time and every haystack as it walks, with the full `CaseFolding.txt` folding `substrings` uses, so `ß` matches `ss`, `ẞ` matches `SS`, and the Kelvin sign matches `k`.
A class holds the folds of its members, and a member folding to several codepoints joins as an alternative tried after the class, so `[ß]` matches `ss` and `[sß]` tries `s` first.
A negated class complements the single-codepoint folds of its members only, so `[^ß]`, `\W` and `.` each match one folded codepoint, and `^.$` does not match `ß`, which folds to two.
Anchors and `\b` are judged on the folded text, so `\b` falls on both sides of `ß` and not between `a` and `ß`.
Lookarounds are judged on the folded text too, but only between whole codepoints: none holds inside the folds of one, and a body ending or starting there marks the edge of the codepoint it snaps out to.

Both start passes step the folds of whole units, right to left, one window at a time rather than eight in lockstep; every byte past ASCII is live, so skips only pass bytes that fold onto themselves.
A start inside the folds of one codepoint marks the codepoint's own start, and the end walk then tries each rune boundary inside its folds in order, the first that ends a match winning.
A match ending inside the folds of a codepoint snaps outward to its end, and the search resumes there, so `s` over `ß` reports the whole codepoint once.
An empty match is never reported inside the folds of a codepoint, as it is never reported before a continuation byte.

## Limits

Each automaton gives way to its fallback past its budget or past 32-bit state ids, so only an NFA past 32-bit state ids is refused with `sz_overflow_risk_k`, the same and only ceiling `substrings` has.
The end-finding automaton is anchored, so it grows linearly with a bounded gap, but not with a gap under an unbounded repetition: `(?:[a-c ]{1,14}.{0,13}\s)*a` would take 1.3 million end-finding states and 4.7 s to compile.
So its subset construction also stops at 16,384 states, and end walks then run a PikeVM over the forward Thompson NFA, with threads in leftmost-first priority over the candidate's span only; that vocabulary compiles in about 45 ms.
No vocabulary is refused for size below the 32-bit ceiling, and device backends run the same fallbacks.

## Methodology

Cells are haystack MB/s over the lines of the first 64 MB of `xlsum.csv`, which the harness's multilingual slice reads.
The vocabularies are StringWars' three, compiled cased: __Literal__ holds `government`, `minister`, `president` and `police`, __Plain__ five everyday patterns for emails, URLs, ISO dates, numbers and the same keywords between word boundaries, and __Lookaround__ four patterns for prices, percentages, names and acronyms, each bounded by a lookbehind or a lookahead.
A `↑` cell reuses the kernel of the tier above, and a `…` cell is not measured yet.

## Counting

Leftmost-first matches per haystack, one count each.

| Backend                       |  Literal |    Plain | Lookaround |
| :---------------------------- | -------: | -------: | ---------: |
| Serial @ 1× Intel Xeon5       |    699.8 |    526.0 |       12.8 |
| Haswell @ 1× Intel Xeon5      |  1,100.8 |    559.8 |       12.8 |
| Skylake @ 1× Intel Xeon5      |        ↑ |        ↑ |          ↑ |
| Ice Lake @ 1× Intel Xeon5     |  1,032.2 |    558.9 |       13.2 |
| CUDA @ 148× Nvidia SM100      | 32,716.8 | 19,384.3 |      425.3 |
| Hopper @ 148× Nvidia SM100    |        ↑ |        ↑ |          ↑ |
| Blackwell @ 148× Nvidia SM100 |        ↑ |        ↑ |          ↑ |

## Locating

Every match, by haystack, pattern and byte span.

| Backend                       |  Literal |    Plain | Lookaround |
| :---------------------------- | -------: | -------: | ---------: |
| Serial @ 1× Intel Xeon5       |    688.7 |    542.4 |       12.4 |
| Haswell @ 1× Intel Xeon5      |  1,073.2 |    548.8 |       12.6 |
| Skylake @ 1× Intel Xeon5      |        ↑ |        ↑ |          ↑ |
| Ice Lake @ 1× Intel Xeon5     |  1,042.4 |    562.3 |       13.0 |
| CUDA @ 148× Nvidia SM100      | 30,617.6 | 17,469.4 |      421.1 |
| Hopper @ 148× Nvidia SM100    |        ↑ |        ↑ |          ↑ |
| Blackwell @ 148× Nvidia SM100 |        ↑ |        ↑ |          ↑ |

## Rewriting

Every match substituted with its pattern's replacement, `<0>` for the first pattern, into one tape.

| Backend                       |  Literal |    Plain | Lookaround |
| :---------------------------- | -------: | -------: | ---------: |
| Serial @ 1× Intel Xeon5       |    701.3 |    526.2 |       12.3 |
| Haswell @ 1× Intel Xeon5      |  1,114.1 |    528.4 |       12.1 |
| Skylake @ 1× Intel Xeon5      |        ↑ |        ↑ |          ↑ |
| Ice Lake @ 1× Intel Xeon5     |  1,010.0 |    543.4 |       12.6 |
| CUDA @ 148× Nvidia SM100      | 22,190.1 | 11,939.8 |      419.8 |
| Hopper @ 148× Nvidia SM100    |        ↑ |        ↑ |          ↑ |
| Blackwell @ 148× Nvidia SM100 |        ↑ |        ↑ |          ↑ |

## Extracting Groups

The year, month and day of every line's first ISO date, as `([0-9]{4})-([0-9]{2})-([0-9]{2})` captures them.

| Backend                       |     Date |
| :---------------------------- | -------: |
| Serial @ 1× Intel Xeon5       |  1,956.9 |
| Haswell @ 1× Intel Xeon5      |  7,653.4 |
| Skylake @ 1× Intel Xeon5      |        ↑ |
| Ice Lake @ 1× Intel Xeon5     |  8,676.4 |
| CUDA @ 148× Nvidia SM100      | 27,166.7 |
| Hopper @ 148× Nvidia SM100    |        ↑ |
| Blackwell @ 148× Nvidia SM100 |        ↑ |

## Compilation

Building both automata is paid once per vocabulary, and it is reported separately because a pipeline that rebuilds per query is bound by this rather than by the walk.
Cells are compilations per second.

| Backend                 | Literal | Plain | Lookaround |
| :---------------------- | ------: | ----: | ---------: |
| Serial @ 1× Intel Xeon5 |   9,644 | 160.2 |    299,200 |
