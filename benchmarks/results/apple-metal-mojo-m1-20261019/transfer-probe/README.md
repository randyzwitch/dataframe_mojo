# Primitive GPU transfer probe: Apple M1, 2026-10-19

The probe produced 30 validated round trips across two sizes, three layouts, and five repetitions. Source commit b2c77a725db56e279fba07bfef67ed00d088aca5 built with Mojo 1.2.0.dev2026100905 (3b71af75) using pixi.lock SHA-256 098d02ba4d166a75640ed6b96aab77fdfe6cb468eded7838f23bba244496e277. Source and executable SHA-256 hashes are in [identity.json](identity.json); all raw stdout is preserved in [samples.txt](samples.txt). The compiler log is empty.

Each table entry is first sample / median of the four later samples, in microseconds. The measured enqueue intervals include host/API enqueue overhead; they are not pure device time. Wait intervals are queue synchronization time. These measurements do not imply copy bandwidth or end-to-end query performance.

| Rows | Layout | Payload / direction | Host alloc | Buffer enqueue | Upload enqueue | Upload wait | Download enqueue | Download wait |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 4,097 | float32_4B | 16,388 B | 1.0 / 1.0 | 17.0 / 16.0 | 272.0 / 259.0 | 239.0 / 223.5 | 321.0 / 354.5 | 239.0 / 244.0 |
| 4,097 | float32_packed_validity | 16,901 B | 5.0 / 1.0 | 28.0 / 27.0 | 530.0 / 531.0 | 253.0 / 275.5 | 647.0 / 644.0 | 281.0 / 259.0 |
| 4,097 | generic_21B_nullable | 86,037 B | 41.0 / 4.5 | 88.0 / 45.5 | 1031.0 / 973.0 | 222.0 / 252.0 | 1460.0 / 1452.0 | 228.0 / 240.5 |
| 1,048,576 | float32_4B | 4,194,304 B | 62.0 / 48.0 | 47.0 / 20.0 | 1174.0 / 711.5 | 245.0 / 218.0 | 13998.0 / 14004.0 | 605.0 / 551.5 |
| 1,048,576 | float32_packed_validity | 4,325,376 B | 71.0 / 49.0 | 21.0 / 23.0 | 1157.0 / 1145.5 | 322.0 / 247.0 | 15012.0 / 14945.5 | 213.0 / 216.5 |
| 1,048,576 | generic_21B_nullable | 22,020,096 B | 4917.0 / 460.5 | 96.0 / 81.0 | 5210.0 / 3731.5 | 253.0 / 214.0 | 79515.0 / 72892.0 | 541.0 / 539.5 |

Generic storage sends separate low-word, high-word, validity, and error buffers (four enqueues in each direction). The high and error planes contain nonzero transport sentinels. Raw Float32 sends one value buffer; the nullable case sends values plus packed validity. Download enqueue time is host-inclusive, especially at 1,048,576 rows, and must not be interpreted as a device-only transfer duration. The rows also reuse a single already-initialized Metal context, so first means first sample for that layout, not a cold-device observation.

Machine-readable phase medians are in [summary.json](summary.json); build identity and hardware are in [identity.json](identity.json).
