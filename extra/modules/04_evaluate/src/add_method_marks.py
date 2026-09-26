"""
Tag the multi-step questions in the answer keys for method marking.

One-shot patch over `04_evaluate/keys/cie*.json`. Idempotent - run it
again and it overwrites the same blocks with the same content.

WHY THESE QUESTIONS AND NOT THE OTHERS
--------------------------------------
Most of this paper is prose, and prose is settled by whether the student
named the right things. A minority of questions are chains: the student
computes a value, then computes the next value from it, and the scheme
awards marks at every link. On those, exact-value checking is unfair in
one specific way - a single wrong figure early makes every later figure
wrong, and an all-or-nothing check charges the same mistake five times.
A human marker charges it once and marks the rest on the student's own
working. That is a "method mark", and it is the one thing neither
keyword coverage nor sentence similarity can produce.

So a question earns a `method` block here only when its later steps
genuinely depend on its earlier ones. CIE-2 3a is the clearest case:
five subnets allocated end to end, where a wrong block size for the
first shifts all of the remaining four. CIE-1 2c - list the five WWW
components - is not, and gets nothing, because its parts are
independent and the keyword tier already settles it.

`steps` is the procedure the marks attach to. `carry_forward` says, in
the examiner's terms, what may be marked on the student's own wrong
value. Both are shipped to the LLM tier in the queue record.
"""

import json
from pathlib import Path

KEYS_DIR = Path(__file__).resolve().parent.parent / "keys"


METHOD = {

    1: {
        "3b": {
            "steps": [
                "Number of unique sequence numbers = 2^m for an m-bit field",
                "The range is therefore 0 to 2^m - 1",
                "Sequence number of packet P = P mod 2^m, counting from 0",
                "Stop-and-Wait uses the range [0,1]; Go-Back-N and "
                "Selective-Repeat with m = 8 use [0,255]",
                "Selective-Repeat send and receive windows are 2^(m-1)",
            ],
            "carry_forward":
                "If the student uses a wrong value for 2^5 or 2^8, mark the "
                "range and the modulo step on their own value.",
        },
        "4b": {
            "steps": [
                "Read each four-hex-digit field of the header in order: "
                "source port, destination port, total length, checksum",
                "Convert each field from hex to decimal",
                "Data length = total length - the 8-byte UDP header",
                "A well-known (low) source port means the sender is a server",
            ],
            "carry_forward":
                "If a hex conversion is wrong, mark the data-length "
                "subtraction and the client/server conclusion on the "
                "student's own converted values.",
        },
    },

    2: {
        "1": {
            "steps": [
                "The host performing the active open is the client; the one "
                "performing the passive open is the server",
                "A SYN consumes one sequence number, so ACK = the received "
                "sequence number + 1",
                "The third segment's sequence number continues from the "
                "client's first segment + 1",
                "The third segment carries the ACK flag only",
            ],
            "carry_forward":
                "If the student misreads a sequence number off the diagram, "
                "mark the + 1 arithmetic and the flag on their own value.",
        },
        "2b": {
            "steps": [
                "For each destination, cost via a neighbour = cost to that "
                "neighbour + the cost that neighbour advertises",
                "Take the minimum across the four neighbours",
                "Record that minimum as the distance and that neighbour as "
                "the next hop",
            ],
            "carry_forward":
                "If one neighbour cost is misread, mark the minimum and the "
                "next hop on the student's own arithmetic for that row. Each "
                "row is judged on its own.",
        },
        "2c": {
            "steps": [
                "Header length = HLEN x 4 bytes",
                "Convert the total length field from hex to decimal",
                "Data carried = total length - header length",
                "Data to fragment = original packet size - header",
                "Maximum data per fragment = MTU - header, rounded DOWN to a "
                "multiple of 8, because offsets are counted in units of 8",
                "The number of fragments follows from dividing the data by "
                "that maximum",
                "Offset = bytes before the fragment / 8; MF = 1 on every "
                "fragment but the last",
            ],
            "carry_forward":
                "Fragment count, sizes and offsets all follow from the "
                "maximum-data-per-fragment figure. If that figure is wrong "
                "but the round-down-to-8 rule was applied, mark everything "
                "after it on the student's own value.",
        },
        "3a": {
            "steps": [
                "Each subnet needs hosts + 2 addresses, for the network and "
                "broadcast addresses",
                "Round that up to the next power of 2 to get the block size; "
                "the mask follows from the block size",
                "Allocate the largest block first, starting at the base "
                "address",
                "Each subnet starts immediately where the previous one ended",
                "First host = network + 1; broadcast = network + block - 1; "
                "last host = broadcast - 1",
            ],
            "carry_forward":
                "The subnets are a chain: a wrong block size for one shifts "
                "every subnet after it. Charge that error once, then mark "
                "each later subnet on the student's own running address - "
                "if their first-host, last-host and broadcast are correct "
                "relative to the network address they themselves arrived "
                "at, that subnet earns its marks.",
        },
        "3b": {
            "steps": [
                "Each organisation needs requested + 2 addresses",
                "Round up to the next power of 2 to get the block size, and "
                "the prefix follows",
                "Allocate sequentially from the base address in the order "
                "requested",
                "Network ID = block start; first usable = +1; broadcast = "
                "block end; last usable = broadcast - 1",
            ],
            "carry_forward":
                "The blocks are allocated end to end, so a wrong block size "
                "for one organisation shifts every organisation after it. "
                "Charge it once and mark the rest on the student's own "
                "running address.",
        },
        "4a": {
            "steps": [
                "Initialise dist(source) = 0 and every other node to infinity",
                "Repeatedly select the unvisited node with the smallest "
                "tentative distance and finalize it",
                "Relax each of its edges: candidate = dist(u) + cost(u,v), "
                "keep it only if smaller than the current distance",
                "Record the predecessor whenever a distance improves",
                "Read the final shortest paths back from the predecessors",
            ],
            "carry_forward":
                "A wrong relaxation early changes every later distance. If "
                "the selection rule and the relaxation rule are applied "
                "correctly from the student's own table, award the later "
                "steps even where the numbers differ from the scheme.",
        },
    },

    3: {
        "2b": {
            "steps": [
                "Bit stuffing: scan left to right and insert a 0 after every "
                "five consecutive 1s",
                "Bit unstuffing: scan for five consecutive 1s followed by a "
                "0, and delete that 0",
            ],
            "carry_forward":
                "Mark the rule and whether it was applied consistently. A "
                "single missed insertion or deletion in a long bit string is "
                "a slip, not a wrong method, and should not void the item.",
        },
        "3a": {
            "steps": [
                "Average signal rate S = c x N x (1/r)",
                "For NRZ-I, c = 1/2 and r = 1, so S = N/2",
                "For Manchester, c = 1 and r = 1/2, so S = N",
                "Minimum bandwidth equals the average signal rate for both "
                "schemes",
            ],
            "carry_forward":
                "If the student uses a wrong c or r, mark the bandwidth step "
                "on the signal rate they themselves computed.",
        },
        "3b": {
            "steps": [
                "Propagation delay = distance / propagation speed",
                "A's first bit reaches B one propagation delay after A starts",
                "B may begin only if it senses the channel idle at that "
                "instant",
                "A learns of the collision one further propagation delay "
                "after B starts",
            ],
            "carry_forward":
                "Both sub-parts hang on the propagation delay. If that "
                "figure is wrong but the idle/busy reasoning is applied "
                "correctly to it, mark both parts on the student's own "
                "value.",
        },
        "4a": {
            "steps": [
                "Transmission delay = packet size in bits / link bandwidth, "
                "for each of the three links",
                "Propagation delay = distance / signal speed, for each link",
                "Round-trip setup = 2 x (sum of transmission + propagation "
                "over all links + both switch processing delays)",
                "Per packet delay = sum of transmission + propagation one way",
                "Total end-to-end = setup + data transfer + teardown",
            ],
            "carry_forward":
                "Every headline figure is built from the per-link delays. If "
                "a per-link delay is wrong but the three sums are formed "
                "correctly, award the setup, per-packet and total steps on "
                "the student's own numbers.",
        },
        "4b": {
            "steps": [
                "r = the degree of the generator, k = dataword length, "
                "n = k + r",
                "Append r zeros to the dataword to form the augmented message",
                "Divide by the generator, XOR-ing at each aligned leading 1",
                "The remainder is the last r bits",
                "Transmitted frame = dataword followed by the remainder",
                "The receiver divides the received frame by the same "
                "generator; a zero remainder means no error",
            ],
            "carry_forward":
                "The frame and the receiver check both follow from the "
                "remainder. If the division goes wrong but the append, the "
                "frame assembly and the verification procedure are right, "
                "award those steps on the student's own remainder.",
        },
    },
}


def main():

    total = 0

    for cie, questions in sorted(METHOD.items()):

        path = KEYS_DIR / f"cie{cie}.json"

        with open(path, encoding="utf-8") as handle:
            key = json.load(handle)

        by_id = {q["id"]: q for q in key["questions"]}

        unknown = set(questions) - set(by_id)
        if unknown:
            raise SystemExit(f"cie{cie}: no such question(s): "
                             f"{', '.join(sorted(unknown))}")

        for qid, block in questions.items():
            by_id[qid]["method"] = block
            total += 1

        # Anything not listed must not keep a stale block from an earlier
        # run - the set of method questions is defined here, not
        # accumulated in the key files.
        for qid, question in by_id.items():
            if qid not in questions:
                question.pop("method", None)

        with open(path, "w", encoding="utf-8") as handle:
            json.dump(key, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

        marks = sum(by_id[q]["marks"] for q in questions)
        print(f"cie{cie}: {len(questions)} method questions "
              f"({', '.join(sorted(questions))}) - {marks} marks")

    print(f"\n{total} of 24 questions tagged for method marking")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
