for i in `find . -name "*.eval" -type f`; do
    perl data/test/eval/wa_eval_align.pl data/test/answers/test.wa.nonullalign $i > "$i.out"
done