# before running this script, use ckpt_zero_to_fp32.sh to convert
# the checkpoint to fp32 transformer format, and then copy the
# config files to the target directory.

BASE="/path/to/source/model"
TARGET="output/models/..."

cp $BASE/config.json                $TARGET/
cp $BASE/generation_config.json     $TARGET/
cp $BASE/preprocessor_config.json   $TARGET/
cp $BASE/tokenizer*                 $TARGET/
cp $BASE/chat_template.json         $TARGET/
