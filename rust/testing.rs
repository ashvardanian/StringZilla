//! Helpers the crate's tests share: the pinned seeds, skips and scratch space, prose fixtures, and
//! the segmenters' batching invariant.
//!
//! Realistic multi-script prose, with ASCII-source `\u{}` escapes and the rendered prose in
//! comments, serves the tests of more than one UTF-8 domain module; per-family segment counts are
//! oracle-locked (ICU root / uniseg). Fixtures read by a single module live beside the test that
//! reads them instead.
//!
//! File: rust/testing.rs
//! Author: Ash Vardanian
//! Date: November 16, 2026

#![allow(dead_code, reason = "the Shared Testing region is the same in every sibling library, used or not")]

extern crate alloc;
extern crate std;

use alloc::vec::Vec;
use std::{eprintln, format};

use crate::{ITERATORS_DEFAULT_STEPS, SegmenterKernel, Utf8Segments};

/// The variable [`seed`] reads, the one the C++ suites read.
const SEED_VARIABLE: &str = "STRINGZILLA_SEED";

// region Shared Testing
/// Announces a skipped test on stderr, so a run shows what it left out rather than passing in
/// silence.
pub(crate) fn skipped(reason: &str) {
    eprintln!("skipped: {reason}");
}

/// The run's seed, read once from the variable [`SEED_VARIABLE`] names: 42 when unset, drawn afresh
/// for `random` and then printed, so the run can be repeated.
pub(crate) fn seed() -> u64 {
    static SEED: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *SEED.get_or_init(read_seed)
}

/// Parses [`SEED_VARIABLE`] once for [`seed`].
fn read_seed() -> u64 {
    use std::hash::{BuildHasher, Hasher};

    const EXPECTED: &str = "an unsigned integer or random";
    let text = std::env::var(SEED_VARIABLE).unwrap_or_default();
    if text.is_empty() {
        return 42;
    }
    if text != "random" {
        let Ok(parsed) = text.parse::<u64>() else {
            panic!("{SEED_VARIABLE}=\"{text}\" does not parse, expected {EXPECTED}");
        };
        return parsed;
    }
    let hasher = std::collections::hash_map::RandomState::new().build_hasher();
    let drawn = hasher.finish();
    eprintln!("{SEED_VARIABLE}={drawn}");
    drawn
}

/// SplitMix64's finalizer, a bijection that spreads every input bit over all 64 output bits.
pub(crate) const fn split_mix_bits(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// The `index`-th stream of the family `name` under `seed`, identical in C, C++, Python and
/// Mojo.
pub(crate) fn stream_key(seed: u64, name: &str, index: u64) -> u64 {
    let mut hashed = 0xCBF2_9CE4_8422_2325_u64;
    for byte in name.bytes() {
        hashed = (hashed ^ u64::from(byte)).wrapping_mul(0x0100_0000_01B3);
    }
    split_mix_bits(split_mix_bits(seed ^ hashed).wrapping_add(index))
}

/// The `index`-th stream of the running test, keyed by the name libtest gives its thread.
pub(crate) fn case_stream_key(index: u64) -> u64 {
    let thread = std::thread::current();
    let name = thread.name().unwrap_or("main");
    stream_key(seed(), name, index)
}

/// A directory of its own under the system's temporary one, removed when dropped.
pub(crate) struct ScratchDirectory {
    path: std::path::PathBuf,
}

impl ScratchDirectory {
    /// Creates `<temp>/<name>-<pid>-<nanos>`, so parallel tests never share one.
    pub(crate) fn new(name: &str) -> Self {
        let since = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH);
        let nanos = since.map_or(0, |elapsed| elapsed.as_nanos());
        let leaf = format!("{name}-{}-{nanos}", std::process::id());
        let path = std::env::temp_dir().join(leaf);
        std::fs::create_dir_all(&path).unwrap();
        Self { path }
    }

    /// Where the directory is.
    pub(crate) fn path(&self) -> &std::path::Path {
        &self.path
    }
}

impl Drop for ScratchDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// Polls `future` to completion on this thread with a no-op waker.
pub(crate) fn block_on<F: core::future::Future>(future: F) -> F::Output {
    let mut future = core::pin::pin!(future);
    let waker = core::task::Waker::noop();
    let mut context = core::task::Context::from_waker(waker);
    loop {
        if let core::task::Poll::Ready(output) = future.as_mut().poll(&mut context) {
            return output;
        }
        std::thread::yield_now();
    }
}
// endregion Shared Testing

/// Segments tile the input, so the yielded segments must match regardless of the batch
/// size `STEPS`; a tiny batch (STEPS == 1) exercises the refill seam on every boundary
/// the kernel reports.
pub(crate) fn assert_steps_invariant<Kernel: SegmenterKernel>(text: &[u8]) {
    let forward: Vec<&[u8]> = Utf8Segments::<Kernel, ITERATORS_DEFAULT_STEPS>::new(text).collect();
    assert_eq!(Utf8Segments::<Kernel, 1>::with_steps(text).collect::<Vec<_>>(), forward);
    assert_eq!(Utf8Segments::<Kernel, 3>::with_steps(text).collect::<Vec<_>>(), forward);
    assert_eq!(Utf8Segments::<Kernel, 65>::with_steps(text).collect::<Vec<_>>(), forward);
}

// Hotel review (German + Japanese): NFD cafe, NBSP-glued units, a sentence-ending abbreviation,
// a CJK run.
pub(crate) const PROSE_HOTEL_REVIEW: &str = concat!(
    "Last spring we strolled down M\u{fc}nchner Stra\u{df}e; the cafe\u{301} cortado cost 3,50\u{a0}",
    "\u{20ac} and was unreal. Dr. Vogel, our guide, swore it's the city's finest. Worth the detour?! ",
    "Absolutely \u{2014} and \u{6771}\u{4eac}\u{30bf}\u{30ef}\u{30fc} the next week, all 333\u{a0}m o",
    "f it, was breathtaking at dusk\u{2026}"
);
// Concert post (Korean + Japanese): conjoining L+V+T jamo, a Katakana run, an ideographic stop,
// a 'p.m.' no-break.
pub(crate) const PROSE_CONCERT_POST: &str = concat!(
    "\u{c624}\u{b298} \u{cf58}\u{c11c}\u{d2b8}, \u{c9c4}\u{c9dc} \u{bbf8}\u{cce4}\u{b2e4}!! \u{1112}",
    "\u{1161}\u{11ab}\u{ad6d} \u{d32c}\u{b4e4}\u{c774} \u{b2e4} \u{baa8}\u{c600}\u{ace0}, the staff b",
    "owed and said \u{c548}\u{b155}\u{d788} \u{ac00}\u{c138}\u{c694}. Setlist was pure \u{30cf}",
    "\u{30fc}\u{30c9}\u{30b3}\u{30a2}; \u{4eca}\u{65e5}\u{306f}\u{6700}\u{9ad8}\u{3060}\u{3063}",
    "\u{305f}\u{3002} We screamed \u{c0ac}\u{b791}\u{d574} till 11 p.m. sharp."
);
// News lede: 'U.S.A.' before a lowercase word without a break, curly quotes, thousands,
// currency, a date range.
pub(crate) const PROSE_NEWS_LEDE: &str = concat!(
    "The U.S.A. wasn't ready, analysts said. \u{201c}We lost 1,000 jobs,\u{201d} the mayor warned. ",
    "\u{201c}Recovery starts now.\u{201d} Filings spiked 2024/06\u{2013}2024/09, topping $1,000 per c",
    "laim. Will it hold?! No one knows for sure."
);
// Language lesson: a Greek final sigma, Cyrillic case pairs, a Croatian titlecase digraph, and
// a fold-only match.
pub(crate) const PROSE_LANGUAGE_LESSON: &str = concat!(
    "Greek lesson: \u{39f}\u{394}\u{39f}\u{3a3} becomes \u{3bf}\u{3b4}\u{3cc}\u{3c2} when lowercased,",
    " ending in a final \u{3c2}. Russian's easy too \u{2014} \u{41c}\u{41e}\u{421}\u{41a}\u{412}",
    "\u{410} \u{2194} \u{43c}\u{43e}\u{441}\u{43a}\u{432}\u{430}, no drama. Croatian has the digraph ",
    "\u{1c4}: titlecase \u{1c5}, lowercase \u{1c6}. Quiz \u{2014} does \u{201c}stra\u{df}e\u{201d} ma",
    "tch STRASSE? Yes, once you fold."
);
// RTL scripts: Hebrew gershayim, Arabic, a number-sign Prepend, an NFC niqqud reorder,
// a Malayalam dot-reph.
pub(crate) const PROSE_RTL_SCRIPTS: &str = concat!(
    "Hebrew acronyms take gershayim: \u{5e6}\u{5d4}\u{5f4}\u{5dc} and \u{5d0}\u{5e8}\u{5d4}\u{5f4}",
    "\u{5d1} aren't typos. Arabic flows right-to-left too \u{2014} \u{645}\u{631}\u{62d}\u{628}",
    "\u{627} \u{628}\u{627}\u{644}\u{639}\u{627}\u{644}\u{645} \u{2014} and finance text can carry th",
    "e number sign \u{600}\u{664}. Niqqud stacks marks: \u{5e9}\u{5c1}\u{5b8}\u{5dc}\u{5d5}\u{5b9}",
    "\u{5dd} must reorder under NFC. Malayalam even has a true prepend, the dot-reph \u{d4e}\u{d15}."
);
