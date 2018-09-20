cur_dir=$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd )

root_dir=$2

cd $root_dir

redo=1
data_root_dir=$1
dataset_name="mit_places"
mapfile="$root_dir/data/$dataset_name/labelmap_oi.prototxt"
anno_type="detection"
db="lmdb"
min_dim=0
max_dim=0
width=0
height=0

extra_cmd="--encode-type=jpg --encoded"
if [ $redo ]
then
  extra_cmd="$extra_cmd --redo --shuffle"
fi
for subset in train val
do
  python $root_dir/scripts/create_annoset.py --anno-type=$anno_type --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height $extra_cmd $data_root_dir $root_dir/data/$dataset_name/$subset.txt $data_root_dir/$db/$dataset_name"_"$subset"_"$db examples/$dataset_name
done
