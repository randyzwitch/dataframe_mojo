"""Code points with the Unicode `Alphabetic` property, for `to_titlecase`.

Title casing capitalizes a letter when the character before it is not
alphabetic. The
ranges above ASCII come from Unicode 15.0 (generated with Perl's
`\\p{Alphabetic}`), as hexadecimal `first-last` pairs.
"""

comptime _RANGES = (
    "aa-aa,b5-b5,ba-ba,c0-d6,d8-f6,f8-2c1,2c6-2d1,2e0-2e4,2ec-2ec,2ee-2ee,"
    "345-345,370-374,376-377,37a-37d,37f-37f,386-386,388-38a,38c-38c,"
    "38e-3a1,3a3-3f5,3f7-481,48a-52f,531-556,559-559,560-588,5b0-5bd,"
    "5bf-5bf,5c1-5c2,5c4-5c5,5c7-5c7,5d0-5ea,5ef-5f2,610-61a,620-657,"
    "659-65f,66e-6d3,6d5-6dc,6e1-6e8,6ed-6ef,6fa-6fc,6ff-6ff,710-73f,"
    "74d-7b1,7ca-7ea,7f4-7f5,7fa-7fa,800-817,81a-82c,840-858,860-86a,"
    "870-887,889-88e,8a0-8c9,8d4-8df,8e3-8e9,8f0-93b,93d-94c,94e-950,"
    "955-963,971-983,985-98c,98f-990,993-9a8,9aa-9b0,9b2-9b2,9b6-9b9,"
    "9bd-9c4,9c7-9c8,9cb-9cc,9ce-9ce,9d7-9d7,9dc-9dd,9df-9e3,9f0-9f1,"
    "9fc-9fc,a01-a03,a05-a0a,a0f-a10,a13-a28,a2a-a30,a32-a33,a35-a36,"
    "a38-a39,a3e-a42,a47-a48,a4b-a4c,a51-a51,a59-a5c,a5e-a5e,a70-a75,"
    "a81-a83,a85-a8d,a8f-a91,a93-aa8,aaa-ab0,ab2-ab3,ab5-ab9,abd-ac5,"
    "ac7-ac9,acb-acc,ad0-ad0,ae0-ae3,af9-afc,b01-b03,b05-b0c,b0f-b10,"
    "b13-b28,b2a-b30,b32-b33,b35-b39,b3d-b44,b47-b48,b4b-b4c,b56-b57,"
    "b5c-b5d,b5f-b63,b71-b71,b82-b83,b85-b8a,b8e-b90,b92-b95,b99-b9a,"
    "b9c-b9c,b9e-b9f,ba3-ba4,ba8-baa,bae-bb9,bbe-bc2,bc6-bc8,bca-bcc,"
    "bd0-bd0,bd7-bd7,c00-c0c,c0e-c10,c12-c28,c2a-c39,c3d-c44,c46-c48,"
    "c4a-c4c,c55-c56,c58-c5a,c5d-c5d,c60-c63,c80-c83,c85-c8c,c8e-c90,"
    "c92-ca8,caa-cb3,cb5-cb9,cbd-cc4,cc6-cc8,cca-ccc,cd5-cd6,cdd-cde,"
    "ce0-ce3,cf1-cf3,d00-d0c,d0e-d10,d12-d3a,d3d-d44,d46-d48,d4a-d4c,"
    "d4e-d4e,d54-d57,d5f-d63,d7a-d7f,d81-d83,d85-d96,d9a-db1,db3-dbb,"
    "dbd-dbd,dc0-dc6,dcf-dd4,dd6-dd6,dd8-ddf,df2-df3,e01-e3a,e40-e46,"
    "e4d-e4d,e81-e82,e84-e84,e86-e8a,e8c-ea3,ea5-ea5,ea7-eb9,ebb-ebd,"
    "ec0-ec4,ec6-ec6,ecd-ecd,edc-edf,f00-f00,f40-f47,f49-f6c,f71-f83,"
    "f88-f97,f99-fbc,1000-1036,1038-1038,103b-103f,1050-108f,109a-109d,"
    "10a0-10c5,10c7-10c7,10cd-10cd,10d0-10fa,10fc-1248,124a-124d,1250-1256,"
    "1258-1258,125a-125d,1260-1288,128a-128d,1290-12b0,12b2-12b5,12b8-12be,"
    "12c0-12c0,12c2-12c5,12c8-12d6,12d8-1310,1312-1315,1318-135a,1380-138f,"
    "13a0-13f5,13f8-13fd,1401-166c,166f-167f,1681-169a,16a0-16ea,16ee-16f8,"
    "1700-1713,171f-1733,1740-1753,1760-176c,176e-1770,1772-1773,1780-17b3,"
    "17b6-17c8,17d7-17d7,17dc-17dc,1820-1878,1880-18aa,18b0-18f5,1900-191e,"
    "1920-192b,1930-1938,1950-196d,1970-1974,1980-19ab,19b0-19c9,1a00-1a1b,"
    "1a20-1a5e,1a61-1a74,1aa7-1aa7,1abf-1ac0,1acc-1ace,1b00-1b33,1b35-1b43,"
    "1b45-1b4c,1b80-1ba9,1bac-1baf,1bba-1be5,1be7-1bf1,1c00-1c36,1c4d-1c4f,"
    "1c5a-1c7d,1c80-1c88,1c90-1cba,1cbd-1cbf,1ce9-1cec,1cee-1cf3,1cf5-1cf6,"
    "1cfa-1cfa,1d00-1dbf,1de7-1df4,1e00-1f15,1f18-1f1d,1f20-1f45,1f48-1f4d,"
    "1f50-1f57,1f59-1f59,1f5b-1f5b,1f5d-1f5d,1f5f-1f7d,1f80-1fb4,1fb6-1fbc,"
    "1fbe-1fbe,1fc2-1fc4,1fc6-1fcc,1fd0-1fd3,1fd6-1fdb,1fe0-1fec,1ff2-1ff4,"
    "1ff6-1ffc,2071-2071,207f-207f,2090-209c,2102-2102,2107-2107,210a-2113,"
    "2115-2115,2119-211d,2124-2124,2126-2126,2128-2128,212a-212d,212f-2139,"
    "213c-213f,2145-2149,214e-214e,2160-2188,24b6-24e9,2c00-2ce4,2ceb-2cee,"
    "2cf2-2cf3,2d00-2d25,2d27-2d27,2d2d-2d2d,2d30-2d67,2d6f-2d6f,2d80-2d96,"
    "2da0-2da6,2da8-2dae,2db0-2db6,2db8-2dbe,2dc0-2dc6,2dc8-2dce,2dd0-2dd6,"
    "2dd8-2dde,2de0-2dff,2e2f-2e2f,3005-3007,3021-3029,3031-3035,3038-303c,"
    "3041-3096,309d-309f,30a1-30fa,30fc-30ff,3105-312f,3131-318e,31a0-31bf,"
    "31f0-31ff,3400-4dbf,4e00-a48c,a4d0-a4fd,a500-a60c,a610-a61f,a62a-a62b,"
    "a640-a66e,a674-a67b,a67f-a6ef,a717-a71f,a722-a788,a78b-a7ca,a7d0-a7d1,"
    "a7d3-a7d3,a7d5-a7d9,a7f2-a805,a807-a827,a840-a873,a880-a8c3,a8c5-a8c5,"
    "a8f2-a8f7,a8fb-a8fb,a8fd-a8ff,a90a-a92a,a930-a952,a960-a97c,a980-a9b2,"
    "a9b4-a9bf,a9cf-a9cf,a9e0-a9ef,a9fa-a9fe,aa00-aa36,aa40-aa4d,aa60-aa76,"
    "aa7a-aabe,aac0-aac0,aac2-aac2,aadb-aadd,aae0-aaef,aaf2-aaf5,ab01-ab06,"
    "ab09-ab0e,ab11-ab16,ab20-ab26,ab28-ab2e,ab30-ab5a,ab5c-ab69,ab70-abea,"
    "ac00-d7a3,d7b0-d7c6,d7cb-d7fb,f900-fa6d,fa70-fad9,fb00-fb06,fb13-fb17,"
    "fb1d-fb28,fb2a-fb36,fb38-fb3c,fb3e-fb3e,fb40-fb41,fb43-fb44,fb46-fbb1,"
    "fbd3-fd3d,fd50-fd8f,fd92-fdc7,fdf0-fdfb,fe70-fe74,fe76-fefc,ff21-ff3a,"
    "ff41-ff5a,ff66-ffbe,ffc2-ffc7,ffca-ffcf,ffd2-ffd7,ffda-ffdc,"
    "10000-1000b,1000d-10026,10028-1003a,1003c-1003d,1003f-1004d,"
    "10050-1005d,10080-100fa,10140-10174,10280-1029c,102a0-102d0,"
    "10300-1031f,1032d-1034a,10350-1037a,10380-1039d,103a0-103c3,"
    "103c8-103cf,103d1-103d5,10400-1049d,104b0-104d3,104d8-104fb,"
    "10500-10527,10530-10563,10570-1057a,1057c-1058a,1058c-10592,"
    "10594-10595,10597-105a1,105a3-105b1,105b3-105b9,105bb-105bc,"
    "10600-10736,10740-10755,10760-10767,10780-10785,10787-107b0,"
    "107b2-107ba,10800-10805,10808-10808,1080a-10835,10837-10838,"
    "1083c-1083c,1083f-10855,10860-10876,10880-1089e,108e0-108f2,"
    "108f4-108f5,10900-10915,10920-10939,10980-109b7,109be-109bf,"
    "10a00-10a03,10a05-10a06,10a0c-10a13,10a15-10a17,10a19-10a35,"
    "10a60-10a7c,10a80-10a9c,10ac0-10ac7,10ac9-10ae4,10b00-10b35,"
    "10b40-10b55,10b60-10b72,10b80-10b91,10c00-10c48,10c80-10cb2,"
    "10cc0-10cf2,10d00-10d27,10e80-10ea9,10eab-10eac,10eb0-10eb1,"
    "10f00-10f1c,10f27-10f27,10f30-10f45,10f70-10f81,10fb0-10fc4,"
    "10fe0-10ff6,11000-11045,11071-11075,11080-110b8,110c2-110c2,"
    "110d0-110e8,11100-11132,11144-11147,11150-11172,11176-11176,"
    "11180-111bf,111c1-111c4,111ce-111cf,111da-111da,111dc-111dc,"
    "11200-11211,11213-11234,11237-11237,1123e-11241,11280-11286,"
    "11288-11288,1128a-1128d,1128f-1129d,1129f-112a8,112b0-112e8,"
    "11300-11303,11305-1130c,1130f-11310,11313-11328,1132a-11330,"
    "11332-11333,11335-11339,1133d-11344,11347-11348,1134b-1134c,"
    "11350-11350,11357-11357,1135d-11363,11400-11441,11443-11445,"
    "11447-1144a,1145f-11461,11480-114c1,114c4-114c5,114c7-114c7,"
    "11580-115b5,115b8-115be,115d8-115dd,11600-1163e,11640-11640,"
    "11644-11644,11680-116b5,116b8-116b8,11700-1171a,1171d-1172a,"
    "11740-11746,11800-11838,118a0-118df,118ff-11906,11909-11909,"
    "1190c-11913,11915-11916,11918-11935,11937-11938,1193b-1193c,"
    "1193f-11942,119a0-119a7,119aa-119d7,119da-119df,119e1-119e1,"
    "119e3-119e4,11a00-11a32,11a35-11a3e,11a50-11a97,11a9d-11a9d,"
    "11ab0-11af8,11c00-11c08,11c0a-11c36,11c38-11c3e,11c40-11c40,"
    "11c72-11c8f,11c92-11ca7,11ca9-11cb6,11d00-11d06,11d08-11d09,"
    "11d0b-11d36,11d3a-11d3a,11d3c-11d3d,11d3f-11d41,11d43-11d43,"
    "11d46-11d47,11d60-11d65,11d67-11d68,11d6a-11d8e,11d90-11d91,"
    "11d93-11d96,11d98-11d98,11ee0-11ef6,11f00-11f10,11f12-11f3a,"
    "11f3e-11f40,11fb0-11fb0,12000-12399,12400-1246e,12480-12543,"
    "12f90-12ff0,13000-1342f,13441-13446,14400-14646,16800-16a38,"
    "16a40-16a5e,16a70-16abe,16ad0-16aed,16b00-16b2f,16b40-16b43,"
    "16b63-16b77,16b7d-16b8f,16e40-16e7f,16f00-16f4a,16f4f-16f87,"
    "16f8f-16f9f,16fe0-16fe1,16fe3-16fe3,16ff0-16ff1,17000-187f7,"
    "18800-18cd5,18d00-18d08,1aff0-1aff3,1aff5-1affb,1affd-1affe,"
    "1b000-1b122,1b132-1b132,1b150-1b152,1b155-1b155,1b164-1b167,"
    "1b170-1b2fb,1bc00-1bc6a,1bc70-1bc7c,1bc80-1bc88,1bc90-1bc99,"
    "1bc9e-1bc9e,1d400-1d454,1d456-1d49c,1d49e-1d49f,1d4a2-1d4a2,"
    "1d4a5-1d4a6,1d4a9-1d4ac,1d4ae-1d4b9,1d4bb-1d4bb,1d4bd-1d4c3,"
    "1d4c5-1d505,1d507-1d50a,1d50d-1d514,1d516-1d51c,1d51e-1d539,"
    "1d53b-1d53e,1d540-1d544,1d546-1d546,1d54a-1d550,1d552-1d6a5,"
    "1d6a8-1d6c0,1d6c2-1d6da,1d6dc-1d6fa,1d6fc-1d714,1d716-1d734,"
    "1d736-1d74e,1d750-1d76e,1d770-1d788,1d78a-1d7a8,1d7aa-1d7c2,"
    "1d7c4-1d7cb,1df00-1df1e,1df25-1df2a,1e000-1e006,1e008-1e018,"
    "1e01b-1e021,1e023-1e024,1e026-1e02a,1e030-1e06d,1e08f-1e08f,"
    "1e100-1e12c,1e137-1e13d,1e14e-1e14e,1e290-1e2ad,1e2c0-1e2eb,"
    "1e4d0-1e4eb,1e7e0-1e7e6,1e7e8-1e7eb,1e7ed-1e7ee,1e7f0-1e7fe,"
    "1e800-1e8c4,1e900-1e943,1e947-1e947,1e94b-1e94b,1ee00-1ee03,"
    "1ee05-1ee1f,1ee21-1ee22,1ee24-1ee24,1ee27-1ee27,1ee29-1ee32,"
    "1ee34-1ee37,1ee39-1ee39,1ee3b-1ee3b,1ee42-1ee42,1ee47-1ee47,"
    "1ee49-1ee49,1ee4b-1ee4b,1ee4d-1ee4f,1ee51-1ee52,1ee54-1ee54,"
    "1ee57-1ee57,1ee59-1ee59,1ee5b-1ee5b,1ee5d-1ee5d,1ee5f-1ee5f,"
    "1ee61-1ee62,1ee64-1ee64,1ee67-1ee6a,1ee6c-1ee72,1ee74-1ee77,"
    "1ee79-1ee7c,1ee7e-1ee7e,1ee80-1ee89,1ee8b-1ee9b,1eea1-1eea3,"
    "1eea5-1eea9,1eeab-1eebb,1f130-1f149,1f150-1f169,1f170-1f189,"
    "20000-2a6df,2a700-2b739,2b740-2b81d,2b820-2cea1,2ceb0-2ebe0,"
    "2f800-2fa1d,30000-3134a,31350-323af,"
)


def alphabetic_ranges() -> List[UInt32]:
    """The ranges above ASCII as first, last pairs, in order."""
    var out = List[UInt32](capacity=1536)
    var value = UInt32(0)
    for byte in StringSlice(_RANGES).as_bytes():
        if byte == 45 or byte == 44:  # '-' or ','
            out.append(value)
            value = 0
        elif byte <= 57:
            value = value * 16 + UInt32(byte - 48)
        else:
            value = value * 16 + UInt32(byte - 87)
    return out^


@always_inline
def is_alphabetic(ranges: List[UInt32], codepoint: UInt32) -> Bool:
    """Whether a code point is alphabetic; `ranges` is alphabetic_ranges()."""
    if codepoint < 128:
        var folded = codepoint | 32
        return folded >= 97 and folded <= 122
    var low = 0
    var high = len(ranges) // 2
    while low < high:
        var middle = (low + high) // 2
        if ranges[2 * middle + 1] < codepoint:
            low = middle + 1
        else:
            high = middle
    return low < len(ranges) // 2 and ranges[2 * low] <= codepoint
