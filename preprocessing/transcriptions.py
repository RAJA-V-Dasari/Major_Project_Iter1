"""
Names and USNs read off each student's cover sheet.

These are HANDWRITTEN fields, so they were transcribed by reading the
contact sheets produced by extract_identity.py rather than by OCR.

`flag` is now empty for every row. All 61 have been read directly off
the cover sheets at full resolution, cross-checked across each
student's CIE covers, and reconciled against the cohort's alphabetical
USN ordering.

student_29 "Pranay Kommuri" was additionally confirmed by the person
himself.

VERIFICATION PASS
-----------------
All 11 originally-flagged rows were re-checked against full-resolution
crops (extract_identity.py --students ... --width 2100). Three
independent signals settled them:

  * USNs in this cohort are assigned ALPHABETICALLY, so a name and its
    number corroborate each other. A reading that breaks the ordering
    is wrong; one that restores it is almost certainly right.
  * Digits were cross-checked against the same writer's `23`/`24` in
    the admission-year field, which is known.
  * Names were compared across all of a student's CIE covers.

Readings that were WRONG and are corrected here:

  student_23  "Aryan Thakur"    -> "Priyam Thakur"
              Decisive: 250 Prerana, 251 Preza, 252 -, 253 Punarnava.
              "Pri..." restores the alphabetical run; "Aryan" broke it.
              The terminal letter shows two arches on both the CIE-2
              and CIE-3 covers, so it is m, not n.

  student_29  "Pranay Kongari"  -> "Pranay Kommuri"
              Double-m clear on the CIE-2 cover; consistent across all
              three. Later confirmed by the person himself.

  student_35  "S. Purni Puvathyusha" -> "S. Punni Prathyusha"
              Two separate errors. The surname is Prathyusha: this
              hand writes r with a pronounced shoulder, plainly there
              after the P, and the earlier reading had inserted a
              spurious u. That same shoulder is ABSENT in the first
              name, so those strokes are nn, not rn - "Punni".

The seven `off-pattern` USNs are all GENUINE, not misreadings. Five of
them form a consecutive block that is also in alphabetical order:

    1BM24CS419  Sachin Kumar E
    1BM24CS420  Sachit P Naidu
    1BM24CS421  Sana T. Pathan
    1BM24CS422  Sandesh
    1BM24CS423  Shravan Shetty

Consecutive *and* alphabetical across five independent hands cannot
arise from misreading. They are a separate 2024-admission group.
1BM23CS372 and 1BM23CS366 were likewise read cleanly off block-capital
covers.

ALPHABETICAL-ORDERING AUDIT
---------------------------
Because this cohort's USNs are alphabetical, sorting the 1BM23CS2xx
rows by number and checking the names are in order audits EVERY row,
not just the ones that looked doubtful. It found two misreadings that
nothing else had flagged - both entries had looked perfectly legible:

  student_39  "Roham Vats" -> "Rohan Vats"
              Predicted from the break at 272/273, then confirmed on
              both covers. Restores Rohan B < Rohan V.

  student_13  1BM23CS228 -> 1BM23CS278
              Predicted from two facts at once: 228 sat far outside
              the 235-290 block, and 278 was a GAP where
              "S. Gajalakshmi" belongs (277 Rukith Nayak, 279 S
              Nafees). Confirmed on all three covers - this writer's
              cursive `2` is curly (compare the `23` year field) while
              the middle digit is angular with a flat top, a `7`.

That is the value of the audit: neither was suspected from the
handwriting, and both were found by the ordering alone.

Four breaks remain and are ARTEFACTS, not errors - near-ties caused by
inconsistent initials and spacing (R. Bhuvan/Rachit,
Raghav/Raghavendra, Rakshit/Rakshitha, S. Purni/Saanvi). Sorting
"Raghav Kaushal" before "Raghavendra Ashok Kumbar" is correct; a
string compare just cannot see it.
"""

# student_id -> (usn, name, flag)
TRANSCRIPTIONS = {
    1: ("1BM23CS245", "Pratheeksha Pai", ""),
    2: ("1BM23CS244", "Pratham K P", ""),
    3: ("1BM23CS243", "Prasobh Ratna Shakya", ""),
    4: ("1BM23CS241", "Pranav Gajanan Kamate", ""),
    5: ("1BM23CS236", "Praagna Dixit", ""),
    6: ("1BM23CS261", "Rakshit S Bhat", ""),
    7: ("1BM24CS420", "Sachit P Naidu", ""),
    8: ("1BM23CS290", "Saksham Shrivastava", ""),
    9: ("1BM23CS286", "Sai Pranav Enjeti", ""),
    10: ("1BM23CS284", "Sahasra Musalikunta", ""),
    11: ("1BM23CS283", "Sagi Vaishnavi", ""),
    12: ("1BM23CS282", "Saanvi S", ""),
    13: ("1BM23CS278", "S. Gajalakshmi", ""),
    14: ("1BM23CS276", "Rudraksh Singh", ""),
    15: ("1BM23CS272", "Rohan B Shekar", ""),
    16: ("1BM23CS269", "Rithvik Kumar R K", ""),
    17: ("1BM23CS264", "Rayhan Sadat Karekal", ""),
    18: ("1BM23CS260", "Rajinder Kumar", ""),
    19: ("1BM23CS259", "Raja Vishwanath Dasari", ""),
    20: ("1BM23CS258", "Raghavendra Ashok Kumbar", ""),
    21: ("1BM23CS255", "Rachit Chandra", ""),
    22: ("1BM23CS253", "Punarnava V", ""),
    23: ("1BM23CS252", "Priyam Thakur", ""),
    24: ("1BM23CS250", "Prerana P Jain", ""),
    25: ("1BM23CS249", "Preetham H D", ""),
    26: ("1BM23CS277", "Rukith Nayak", ""),
    27: ("1BM23CS247", "Prathith Suresh Rao", ""),
    28: ("1BM23CS246", "Pratheeth S Angadi", ""),
    29: ("1BM23CS242", "Pranay Kommuri", ""),
    30: ("1BM23CS239", "Prakrithi Jain", ""),
    31: ("1BM23CS235", "P. Ashrita", ""),
    32: ("1BM23CS288", "Saksham Pandey", ""),
    33: ("1BM23CS287", "Saketh Bharadwaj K H", ""),
    34: ("1BM23CS285", "Sahil Sharma", ""),
    35: ("1BM23CS281", "S. Punni Prathyusha", ""),
    36: ("1BM23CS280", "S Nagashree", ""),
    37: ("1BM23CS279", "S Nafees", ""),
    38: ("1BM23CS275", "Roshan N", ""),
    39: ("1BM23CS273", "Rohan Vats", ""),
    40: ("1BM23CS271", "Ritu Sinha", ""),
    41: ("1BM23CS270", "Rithvik N", ""),
    42: ("1BM23CS268", "Rithika R", ""),
    43: ("1BM23CS267", "Rishab M Jain", ""),
    44: ("1BM23CS266", "Ridhima Suhane", ""),
    45: ("1BM23CS265", "Revathi R", ""),
    46: ("1BM23CS263", "Rashi K S", ""),
    47: ("1BM23CS262", "Rakshitha Yathiraj", ""),
    48: ("1BM23CS257", "Raghav Kaushal", ""),
    49: ("1BM23CS256", "Rachit Sinha", ""),
    50: ("1BM23CS251", "Preza Mishra", ""),
    51: ("1BM23CS248", "Praveen Rathod", ""),
    52: ("1BM23CS254", "R. Bhuvan Aditya", ""),
    53: ("1BM23CS240", "Pramitha J O", ""),
    54: ("1BM23CS237", "Pradhan Sagar K", ""),
    55: ("1BM23CS372", "Samartha S R", ""),
    56: ("1BM23CS366", "Bhavana S Holla", ""),
    57: ("1BM24CS423", "Shravan Shetty", ""),
    58: ("1BM24CS422", "Sandesh", ""),
    59: ("1BM24CS419", "Sachin Kumar E", ""),
    60: ("1BM24CS421", "Sana T. Pathan", ""),
    61: ("1BM23CS289", "Saksham Raj", ""),
}
