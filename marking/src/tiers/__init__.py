"""The marking ladder: exact values, then similarity, then a model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Thresholds:
    """Where each tier's reliable region ends.

    These are parameters, not constants, for one reason: they have to be
    swept against the examiner's marks in calibrate.py. If the ladder
    read them from module globals, calibration would need its own copy
    of the decision logic, and the two copies would drift apart on the
    first change - leaving a "calibrated" number that describes code
    nobody runs.

    THESE ARE NOT THE SWEEP'S PICK - AN OPEN DEVIATION
    --------------------------------------------------
    `calibrate.py` sweeps ~1,600 settings and marks its recommendation in
    output/calibration.md. It currently recommends kw_confident 0.85 and
    kw_corroborate 1.00, and these values are looser than that. Measured
    on the same basis, over the ladder alone:

        shipped       over 11, under 5, 5.5% error, 22% decisive
        sweep's pick  over 10, under 5, 5.2% error, 20% decisive

    So shipped buys two points of decisiveness with one extra
    over-settled question, and this project's own selection rule -
    fewest irreversible errors first, decisiveness only as a tie-break -
    says that is the wrong trade. Decisiveness saves human time; an
    over-settled mark awards credit the student did not earn and no
    later tier can take it back.

    It is left here rather than quietly changed because moving it
    re-marks fifty students' work, and because kw_corroborate 1.00 means
    similarity may only confirm an item whose keywords are ALL present,
    which retires the semantic tier in all but name. That is a decision
    to take deliberately, not a constant to nudge. Until it is taken,
    the deviation is recorded here rather than hidden.

    `zero_max_chars = 150` needs no such argument: the sweep agrees.

    The numbers above are the only ones quoted in this file, and they are
    a comparison the reports do not print. Everything else lives in the
    report that generates it:

        output/calibration.md   what the sweep recommends
        output/agreement.md     what the shipped settings actually do

    An earlier version of this docstring said the sweep had CHOSEN these
    values. It had not, and the claim survived because nothing re-checked
    it against a report that was itself months stale.

    The selection rule in that sweep is "fewest irreversible errors, then
    most decisive" - not "most decisive within some error budget".
    Settling more marks only saves human time, while an over-settled mark
    awards credit the student did not earn and no later tier can take it
    back. With the model tier running free on a borrowed GPU, trading
    decisiveness for fewer irreversible errors costs nothing that
    matters.
    """

    # Keyword coverage above which the student named essentially
    # everything the item asks for, and below which almost none of it.
    kw_confident: float = 0.67
    kw_absent: float = 0.34

    # Similarity may only confirm an item with at least this much
    # keyword support. This is a rule rather than a knob - see
    # semantic.py before moving it - but it lives here, and only here,
    # because calibrate.py has to be able to sweep it. semantic.py kept
    # its own stale copy of this number for a while, saying 0.50 while
    # the tier ran on 0.67.
    kw_corroborate: float = 0.67

    # Cosine similarity above which the claim is being made in other
    # words.
    sem_confident: float = 0.62

    # A cheap tier may record a ZERO only when the answer is at most
    # this long. Past it, zero keyword overlap is far more likely to
    # mean our vocabulary missed theirs than that the student said
    # nothing.
    #
    # Measured on CIE-2 2a: seven students wrote 270-654 characters that
    # the examiner accepted, and our keywords scored 0.00 on them
    # because they described a lost acknowledgment instead of naming the
    # RTO. Zeroing those is irreversible and wrong. Set very large to
    # allow zeroing at any length; calibrate.py sweeps it.
    zero_max_chars: int = 150


DEFAULT = Thresholds()
