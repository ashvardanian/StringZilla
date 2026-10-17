# Subdocuments: Path Queries Without a Tree

This directory holds the kernels behind `sz_subdocuments_engine_init`, `sz_subdocuments_counts` and `sz_subdocuments_find`.
One engine owns a compiled vocabulary of paths and the depth stack its walks grow, and pulls every value those paths select out of a batch of documents in one pass per document, whatever the path count.
A match is the subdocument's exact bytes inside its document: in JSON a string with its quotes, a number or literal without the blank space around it, or an object or array through its closing bracket; in XML an element, an attribute's value or a text node, as [XML and XPath](#xml-and-xpath) defines them.
Nothing is copied or decoded, so a caller reads a match in place, hands it to another parser, or unescapes a string body with `sz_strings_unescape_best` and `sz_escape_json_k`.
JSON documents follow RFC 8259 and paths follow RFC 9535: the root `$`, child and descendant segments, names, wildcards, indices and slices, negative ones counting from the array's end, filters, and unions of them in brackets.
XML documents follow XML 1.0 and paths XPath 1.0 along forward axes.

## The Step Automaton

A path compiles into __steps__, one per segment, numbered path after path, the first step of every path being its root `$`.
A bracketed union `[a,'b',1:3,*]` is one step whose selection is the OR of its selectors, so a node is reported once per path however many selectors reach it.
"Live" is a bitset of steps, so a value's live set says which prefixes of which paths reach it, and every path advances together in a few word operations.
Entering a member with key `k` or an element at position `n` computes the child's set from its parent's:

```text
live' = ((live & ~final_steps) << 1 & selected) | (live & descendant_steps)
```

The shift moves every live step that is not the last of its path to the next one, and `selected` keeps those whose selectors accept the child: the steps naming `k`, the wildcards, and the steps with a slice containing `n`.
A descendant segment `..s` applies `s` to a node and to all its descendants, so the step before it stays live at every depth below, which is all `descendant_steps` holds.
A value whose set holds a final step is a match, once per path, however many routes reach it; matches come in document order, by path within one node, so a container precedes what it holds.
A key is looked up once per member in an open-addressed table of every name the paths spell, hashed with one multiply and shift per 8 bytes, and confirmed against the stored bytes; a key holding escapes is unescaped first, and a key longer than every name matches none.
Keys are short and many: `sz_hash`, whose AES rounds the serial tier emulates, would take a third of the NEON walk's time.
So the hash and the comparisons read a word at a time, a key no name matches in length is never hashed, and a key the classified block holds whole is known free of escapes from the block's mask of backslashes.
Since the set of a node depends only on its ancestors' keys and positions, the walk needs one live set per open container and nothing else, kept on a stack that grows on demand.
A vocabulary of at most 64 steps, nearly every one, runs a walk specialized for a single word, where no loop over words remains.

## Pending Runs

Negative indices, negative bounds and steps below -1 select positions that depend on the array's length, which only its closing bracket settles.
Such steps form `relative_steps`, and an array whose live set advances into one opens a __pending run__.
Inside it the walk carries a tentative live set, in which every relative step selects every element; the transition is monotone in its selections, so the tentative set holds every step the true one does.
Matches are not reported then; instead the depth stack stops freeing slots, and keeps every container entered and every node the tentative set would report, each with its parent's slot, its key's name or its position, and its byte span, in document order.
When the run's array closes, every array inside it has closed too, so every length is known, and a replay walks the kept slots in order, recomputes each node's live set from its parent's with the true selections, and reports the matches.
They come out in document order without a sort, and no byte is read twice.

Tagging matches with the condition they wait on would not be enough: with descendant segments and nested arrays a match can wait on several conditions at once, joined by OR across its routes, which the replay settles by recomputing the live sets instead.

## Filters

A filter selector `[?…]` keeps the children for which its expression holds: comparisons, existence tests of relative queries, `&&`, `||`, `!` and parentheses, and the functions `length()`, `count()`, `value()`, `match()` and `search()`.
Each filter compiles into a postfix program, and each query it holds, `@` followed by any segments, into a path of its own, whose first step stands for the candidate child and whose last marks what the query selects.
A path is parsed twice: once to number its own steps, skipping filter bodies, then again to compile its filters, whose queries take steps after every step numbered so far, so the steps of every path stay contiguous whatever its filters nest.

A child's selection waits until the child closes, so a filter step works like a relative slice: a container whose live set advances into one opens a pending run.
The walk selects every candidate tentatively and makes the first steps of its filters' queries live there, so it keeps every node a filter may read.
The replay settles a candidate before the nodes inside it: each filter replays the candidate's subtree from its queries' first steps, the nodes where a query's last step is live are what the query selects, and the program runs on their values.
A nested filter replays inside its query's replay, and a subtree whose live set is empty is skipped, so a filter costs the nodes its queries reach.
Every kept node's subtree ends where the next node outside it starts, which one backward pass over the run settles before the replay.

Comparisons follow RFC 9535: Nothing equals only Nothing, numbers compare as numbers, strings by codepoints, and values of different kinds are only unequal.
One side must be a literal, `length()` or `count()`, so no comparison needs the deep equality of two document values; absolute queries inside filters are refused too, as the document's root may come after the candidate closes.
`match()` and `search()` take I-Regexp patterns as string literals, translated so `.` skips line breaks and `^` and `$` stay characters, and compiled together into one regex engine.
Leftmost-first matching across patterns would let one pattern hide another's match, so each pattern is compiled behind a tag naming it, `\A\x01<index>\x02`, and a tested string carries the tag of the pattern it is tested against.
That engine is compiled through the regex family's dispatch, which a header-only build does not link.

## The Block Masks

The walk never looks at most bytes.
Each 64-byte block is classified into bitmasks first, one bit per byte, and the walk then visits only the structural bits in order with `sz_u64_ctz`.
Quotes, backslashes, brackets, colons, commas and blank space each get a mask.
A quote is escaped when an odd run of backslashes precedes it, which one carry-propagating subtraction finds for the whole block at once.
The in-string mask is the prefix XOR of the remaining quotes, computed with a six-step shift ladder, and every bracket, colon and comma inside a string is cleared with it.
The first byte of each run of scalar bytes, outside strings and neither blank nor structural, is marked too, so numbers, `true`, `false` and `null` are visited once each.
The run's end is the first byte the block's mask of scalar bytes leaves out, and an integer of at most 16 bytes is checked by two word tests for digits, with no loop per byte.
Three carries cross from block to block: a pending escape, being inside a string, and being inside a scalar.
The serial tier builds these masks eight bytes at a time with SWAR compares, and a SIMD tier replaces only that classifier and the skim of [Skipping](#skipping), leaving the walk shared.

Most SIMD tiers classify with one pair of nibble lookups per byte: the entries of the low and high nibble are ANDed, and each of eight bits marks a class, `{[`, `}]`, `:`, `,`, `"`, `\`, space, and the other blanks.
Every class is a product of a set of low nibbles and a set of high nibbles, so no other byte gets a bit.
`/` alone gets two, the comma's and the space's, so the settle finds every slash outside strings without a ninth class.
On Ice Lake each class mask is then one byte test into a mask register.
Haswell, WASM, LASX and VSX shift each class bit to the top of its byte instead, so a class mask is one movemask per register.
NEON has no such test, and one movemask per class, four tests and a tree of pairwise additions each, classifies at 8.0 GB/s on M5.
Instead, the block is loaded deinterleaved, so lane `l` of vector `k` holds byte `4l + k`, and one 8×8 bit-plane transpose turns the 64 class bytes into all eight masks: two rounds of masked shifts between vectors gather each class's four bits into a nibble, and neighbouring lanes' nibbles are whole mask bytes.
That runs at 13.3 GB/s with the same masks.
SVE2 needs no table: MATCH tests every byte against a set its 128-bit segment holds, one instruction per class, and a carry-free multiply packs each 64-bit lane's predicate bytes before one OR reduction per mask, at any vector length.
RVV compares too, since a lookup across the four registers a 64-byte block takes costs a permutation that grows with the square of the group, and each mask register reads out as a mask in one move.

## Skipping

A container whose live set holds no step that can advance is never entered.
Its end is found by counting brackets: while a block's closing brackets cannot outnumber the depth, the depth just moves by the difference of the opening and closing popcounts, and only the block where it may reach zero is walked bit by bit.
The blocks before that one are only skimmed: a tier's skim masks quotes, backslashes, brackets and slashes, five classes instead of the walk's six, and the block where the depth may reach zero is classified again, in full, from the carries it started with.
A tier without a skim of its own hands its classifier instead.
Strings cost nothing extra either way, since the walk's next structural byte after an opening quote is always the closing one.
On a corpus where a path touches a handful of values per document, nearly all of the work is the classification itself.

## Validation

A document is checked where the walk reads it: bracket kinds, the order of keys, colons, values and commas, scalar grammar, unescaped keys it compares, and blank space after the root.
Every byte is classified, so an unterminated string, unbalanced brackets or a `/` outside strings fail the call wherever they are, with `sz_invalid_syntax_k` and the first such document's index.
No JSON document holds such a slash, and every comment of the lenient dialects starts with one, so a comment never hides brackets or members inside a skipped container.
A skipped container is otherwise only counted, so a malformed scalar or mismatched bracket kinds inside it go unnoticed, and string bodies are never validated beyond the escapes of a compared key.

## XML and XPath

An engine compiled with `sz_subdocuments_xml_k` reads XML 1.0 and takes XPath 1.0 location paths along forward axes: `/` and `//` steps, each a name test, `*`, `@name`, `@*` or `text()`, with an optional position `[n]`.
The same automaton runs on them: `//` is a descendant segment, `attribute_steps` and `text_steps` keep `@` and `text()` steps to their node kinds, and the document node holds the root steps, its one element child being the root element.
Name tests compare QNames as the document spells them, `p:item` matching `<p:item>` whatever namespace `p` is bound to, and `xmlns` declarations are no attributes, as in XPath's data model.
A position counts the siblings passing the same test, so `//a[1]` is every `a` that comes first among its parent's `a` children; each open element keeps one counter per positional step, after its live set, indexed by the step's only slice.

A match spans an element from `<` through its end tag or `/>`, an attribute's value without its quotes, or a text node.
A text node is any run of character data and CDATA sections between other markup, so comments and processing instructions split one and CDATA sections join it, an empty CDATA section included, as libxml2 builds them.

XML lexing needs no state across blocks: whether a quote matters depends on being inside a tag, which the lexer already knows.
So the walk classifies nothing; it runs the tier's byte search to the one byte its state needs, `<` in character data, the closing quote of an attribute value, `>` in comments, CDATA sections and processing instructions, and reads names byte by byte.
On M5, a NEON block classifier of `<`, `>`, `"` and `'` built on the JSON transpose runs the walk at 5.9 to 8.1 GB/s, and the byte search, about five instructions per 16 bytes of character data, at 8.5 to 10.7 GB/s; the serial tier's SWAR search runs it at 6.7 to 8.6 GB/s, against 3.0 to 3.6 for SWAR block masks.

An element no live step can advance from is lexed to its end tag without a slot; its own end tag must repeat its name, while what it holds is checked only for nesting.
A DOCTYPE is skipped when it has no internal subset, and refused when it has one, since the entities it may declare are never expanded.

## Why No Tree

A tape or a DOM materializes structure that a query then navigates, and most of that structure belongs to values no path selects.
Here the paths are known before the documents, so the automaton decides at each container whether anything below it can matter, and the bytes of everything that cannot are classified once and never revisited.
The same shape carries to devices, where classification is a prefix XOR and depth a prefix sum, and the live sets advance one depth level at a time.

## Methodology

Cells are document MB/s over one document per line: the JSON events of the first 64 MB of `gharchive/2024-01-01-15.json`, and the `<row>` elements of the first 16 MB of `stackexchange-bioinformatics/Posts.xml`.
A line that is not a whole document once trimmed, like the dump's header, is left out, as StringWars leaves it out.
The vocabularies are StringWars': __Pointers__ are `$.type`, `$.actor.login` and `$.repo.name`, the paths a JSON pointer spells, and __Queries__ are `$.payload.commits[*].sha`, `$..url` and `$.payload.commits[?@.distinct == true].message`.
A `↑` cell reuses the kernel of the tier above, and a `…` cell is not measured yet.

## JSONPath

| Backend                       | Count, Pointers | Count, Queries | Find, Pointers | Find, Queries |
| :---------------------------- | --------------: | -------------: | -------------: | ------------: |
| Serial @ 1× Intel Xeon5       |           894.4 |          607.7 |          877.6 |         596.0 |
| Haswell @ 1× Intel Xeon5      |         3,647.5 |        1,199.1 |        3,626.0 |       1,186.8 |
| Skylake @ 1× Intel Xeon5      |               ↑ |              ↑ |              ↑ |             ↑ |
| Ice Lake @ 1× Intel Xeon5     |         3,180.5 |        1,153.0 |        3,705.9 |       1,185.8 |
| CUDA @ 148× Nvidia SM100      |        49,879.0 |        5,995.5 |       24,217.6 |       3,049.5 |
| Hopper @ 148× Nvidia SM100    |               ↑ |              ↑ |              ↑ |             ↑ |
| Blackwell @ 148× Nvidia SM100 |               ↑ |              ↑ |              ↑ |             ↑ |

## XPath

`/row/@Id`, `/row/@Score`, `/row/@Tags` and `/row/@OwnerUserId` over every row.

| Backend                       |    Count |    Find |
| :---------------------------- | -------: | ------: |
| Serial @ 1× Intel Xeon5       |  3,488.8 | 3,204.1 |
| Haswell @ 1× Intel Xeon5      |  4,647.9 | 3,661.8 |
| Skylake @ 1× Intel Xeon5      |        ↑ |       ↑ |
| Ice Lake @ 1× Intel Xeon5     |  4,758.5 | 4,081.7 |
| CUDA @ 148× Nvidia SM100      | 11,059.2 | 5,334.0 |
| Hopper @ 148× Nvidia SM100    |        ↑ |       ↑ |
| Blackwell @ 148× Nvidia SM100 |        ↑ |       ↑ |
