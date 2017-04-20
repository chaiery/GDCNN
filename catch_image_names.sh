for i in `ls ./NewPNGlabeled`
do 
	bname=$(basename $i)
	echo $bname >> image_collection
done