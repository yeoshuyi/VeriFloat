/* Checks a flat corpus (tools/golden_corpus flat) through libverifloat's C
   API and nothing else: no Python, no C++ standard library on this side.
   Meant for platforms the Python test suite does not reach.

       corpus_check corpus.tsv

   Each line: kind p1 p2 p3 p4 <tab> format <tab> target format <tab>
   sr a b c result flags (hex). Prints the number of vectors checked,
   skipped (what the C API has no entry point for) and mismatched; exit
   status 1 if any differ. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "verifloat.h"

#define WORDS 16

static void parse(const char *s, uint32_t *w) {      /* hex of any length into 32-bit words */
    size_t n = strlen(s);
    memset(w, 0, WORDS * sizeof *w);
    for (size_t i = 0; i < n && i / 8 < WORDS; ++i) {
        char ch = s[n - 1 - i];
        uint32_t d = ch <= '9' ? (uint32_t)(ch - '0') : (uint32_t)((ch | 32) - 'a' + 10);
        w[i / 8] |= d << (4 * (i % 8));
    }
}

static uint64_t low64(const uint32_t *w) { return (uint64_t)w[0] | ((uint64_t)w[1] << 32); }

static char *field(char **p) {                         /* next tab-separated field */
    char *start = *p, *end = strpbrk(start, "\t\r\n");
    if (end) {
        *end = 0;
        *p = end + 1;
    } else {
        *p = start + strlen(start);
    }
    return start;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: corpus_check corpus.tsv\n");
        return 2;
    }
    FILE *fh = fopen(argv[1], "r");
    if (!fh) {
        perror(argv[1]);
        return 2;
    }
    static char line[4096];
    long checked = 0, skipped = 0, bad = 0, n = 0;
    while (fgets(line, sizeof line, fh)) {
        ++n;
        char *p = line;
        int kind = atoi(field(&p)), q[4];
        for (int i = 0; i < 4; ++i) q[i] = atoi(field(&p));
        const vf_fmt *f = vf_format(field(&p)), *t = vf_format(field(&p));
        if (!f || !t) {
            printf("line %ld: format not accepted: %s\n", n, vf_last_error());
            ++bad;
            continue;
        }
        uint32_t sr[WORDS], a[WORDS], b[WORDS], c[WORDS], want[WORDS], got[WORDS] = {0};
        parse(field(&p), sr);
        parse(field(&p), a);
        parse(field(&p), b);
        parse(field(&p), c);
        parse(field(&p), want);
        int want_flags = (int)strtol(field(&p), NULL, 16), flags = 0;
        int wide = vf_format_size(f) > 64, wide_t = vf_format_size(t) > 64;
        int words = (vf_format_size(t) + 31) / 32;
        uint64_t r = 0;
        vf_set_sr(low64(sr));
        switch (kind) {
            case 0:
                if (wide) vf_op_w(f, q[0], a, b, c, got, &flags);
                else r = vf_op(f, q[0], low64(a), low64(b), low64(c), &flags);
                break;
            case 1:
                words = 1;
                if (wide && q[0] != 0) { ++skipped; continue; }       /* only vf_compare has a wide form */
                if (wide) r = (uint64_t)vf_compare_w(f, a, b, q[1], &flags);
                else if (q[0] == 0) r = (uint64_t)vf_compare(f, low64(a), low64(b), q[1], &flags);
                else if (q[0] == 1) r = (uint64_t)vf_eq(f, low64(a), low64(b), q[1], &flags);
                else if (q[0] == 2) r = (uint64_t)vf_lt(f, low64(a), low64(b), q[1], &flags);
                else r = (uint64_t)vf_le(f, low64(a), low64(b), q[1], &flags);
                wide = 0;
                break;
            case 2:
                if (wide || wide_t) {
                    vf_convert_w(t, f, a, got, &flags);
                    wide = 1;
                } else r = vf_convert(t, f, low64(a), &flags);
                break;
            case 3:
                if (wide) vf_round_to_integral_w(f, a, q[0], q[1], got, &flags);
                else r = vf_round_to_integral(f, low64(a), q[0], q[1], &flags);
                break;
            case 4:
                words = 2;
                if (wide) { ++skipped; continue; }
                r = vf_to_int(f, low64(a), q[0], q[1], q[2], q[3], &flags);
                break;
            default:
                if (wide) { ++skipped; continue; }
                r = q[1] ? vf_from_int(f, (int64_t)low64(a), &flags) : vf_from_uint(f, low64(a), &flags);
                break;
        }
        if (!wide) {
            got[0] = (uint32_t)r;
            got[1] = (uint32_t)(r >> 32);
        }
        ++checked;
        if (flags != want_flags || memcmp(got, want, (size_t)(words < 2 ? 2 : words) * sizeof *got)) {
            if (++bad <= 10)
                printf("line %ld: kind %d: got %08x%08x flags %x, expected %08x%08x flags %x (%s)\n", n, kind,
                       got[1], got[0], flags, want[1], want[0], want_flags, vf_last_error());
        }
    }
    fclose(fh);
    printf("libverifloat %s: %ld vectors checked, %ld skipped, %ld mismatches\n", vf_version(), checked, skipped, bad);
    puts(bad ? "DIFFERENT" : "IDENTICAL");
    return bad ? 1 : 0;
}
