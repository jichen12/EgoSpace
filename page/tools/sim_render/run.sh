O=/tmp/egospace_sim; rm -rf $O; mkdir -p $O
PY=/home/jc/jc/spider/.venv/bin/python
for spec in "clip-002140 0.6 0.6" "clip-002301 0.6 0.9"; do
 set -- $spec
 for m in gt_labels infer_ours egohoi; do pat=$1-0-80-obj; [ $m = gt_labels ] && pat=$1-obj
  d=$(ls -d /home/jc/jc/outputs/spider_corrected_v3_$m/processed/hot3d/xhand/*/${pat}* | head -1)
  dist=$2; [ $m = egohoi ] && dist=$3
  LOOK=right_object,right_hand_link,left_hand_link DIST=$dist MUJOCO_GL=osmesa W=640 H=640 $PY /tmp/egospace_render_seq.py $d $O/${1}_0-80/$m 7 > $O/${1}_$m.log 2>&1 &
 done
done
wait; echo alldone
