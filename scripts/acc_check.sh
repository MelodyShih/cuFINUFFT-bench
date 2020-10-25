## Type 1
N1=$1
N2=$N1
N3=1
DIM=$2

if [ $DIM == 2 ]
then
	N3=1
else
	N3=$N1
fi
M=$(echo $N1*$N2*$N3 | bc)

echo lib type N1 N2 N3 M w reltol
# FINUFFT
OUT=$(../finufft_type1 1 $DIM $N1 $N2 $N3 $M 1e-1)
acc=$(echo $OUT | grep -o -P '(?<=F is).*')
ns=$(echo $OUT | grep -o -P '(?<=ns=).*?(?= (\[time))')
echo 0 1 $N1 $N2 $N3 $M $ns $acc
OUT=$(../finufft_type2 1 $DIM $N1 $N2 $N3 $M 1e-1)
acc=$(echo $OUT | grep -o -P '(?<=c is).*')
ns=$(echo $OUT | grep -o -P '(?<=ns=).*?(?= (\[time))')
echo 0 2 $N1 $N2 $N3 $M $ns $acc

OUT=$(../finufft_type1 1 $DIM $N1 $N2 $N3 $M 1e-3)
acc=$(echo $OUT | grep -o -P '(?<=F is).*')
ns=$(echo $OUT | grep -o -P '(?<=ns=).*?(?= (\[time))')
echo 0 1 $N1 $N2 $N3 $M $ns $acc
OUT=$(../finufft_type2 1 $DIM $N1 $N2 $N3 $M 1e-3)
acc=$(echo $OUT | grep -o -P '(?<=c is).*')
ns=$(echo $OUT | grep -o -P '(?<=ns=).*?(?= (\[time))')
echo 0 2 $N1 $N2 $N3 $M $ns $acc

OUT=$(../finufft_type1 1 $DIM $N1 $N2 $N3 $M 1e-5)
acc=$(echo $OUT | grep -o -P '(?<=F is).*')
ns=$(echo $OUT | grep -o -P '(?<=ns=).*?(?= (\[time))')
echo 0 1 $N1 $N2 $N3 $M $ns $acc
OUT=$(../finufft_type2 1 $DIM $N1 $N2 $N3 $M 1e-5)
acc=$(echo $OUT | grep -o -P '(?<=c is).*')
ns=$(echo $OUT | grep -o -P '(?<=ns=).*?(?= (\[time))')
echo 0 2 $N1 $N2 $N3 $M $ns $acc

OUT=$(../finufft_type1 1 $DIM $N1 $N2 $N3 $M 1e-7)
acc=$(echo $OUT | grep -o -P '(?<=F is).*')
ns=$(echo $OUT | grep -o -P '(?<=ns=).*?(?= (\[time))')
echo 0 1 $N1 $N2 $N3 $M $ns $acc
OUT=$(../finufft_type2 1 $DIM $N1 $N2 $N3 $M 1e-7)
acc=$(echo $OUT | grep -o -P '(?<=c is).*')
ns=$(echo $OUT | grep -o -P '(?<=ns=).*?(?= (\[time))')
echo 0 2 $N1 $N2 $N3 $M $ns $acc

# CUFINUFFT
OUT=$(../cufinufft_type1 1 $DIM $N1 $N2 $N3 $M 1e-0)
acc=$(echo $OUT | grep -o -P '(?<=F is).*')
ns=$(echo $OUT | grep -o -P '(?<=ns=).*?(?= (\[time))')
echo 1 1 $N1 $N2 $N3 $M $ns $acc
OUT=$(../cufinufft_type2 1 $DIM $N1 $N2 $N3 $M 1e-0)
acc=$(echo $OUT | grep -o -P '(?<=c is).*')
ns=$(echo $OUT | grep -o -P '(?<=ns=).*?(?= (\[time))')
echo 1 2 $N1 $N2 $N3 $M $ns $acc

OUT=$(../cufinufft_type1 1 $DIM $N1 $N2 $N3 $M 1e-2)
acc=$(echo $OUT | grep -o -P '(?<=F is).*')
ns=$(echo $OUT | grep -o -P '(?<=ns=).*?(?= (\[time))')
echo 1 1 $N1 $N2 $N3 $M $ns $acc
OUT=$(../cufinufft_type2 1 $DIM $N1 $N2 $N3 $M 1e-2)
acc=$(echo $OUT | grep -o -P '(?<=c is).*')
ns=$(echo $OUT | grep -o -P '(?<=ns=).*?(?= (\[time))')
echo 1 2 $N1 $N2 $N3 $M $ns $acc

OUT=$(../cufinufft_type1 1 $DIM $N1 $N2 $N3 $M 1e-4)
acc=$(echo $OUT | grep -o -P '(?<=F is).*')
ns=$(echo $OUT | grep -o -P '(?<=ns=).*?(?= (\[time))')
echo 1 1 $N1 $N2 $N3 $M $ns $acc
OUT=$(../cufinufft_type2 1 $DIM $N1 $N2 $N3 $M 1e-4)
acc=$(echo $OUT | grep -o -P '(?<=c is).*')
ns=$(echo $OUT | grep -o -P '(?<=ns=).*?(?= (\[time))')
echo 1 2 $N1 $N2 $N3 $M $ns $acc

OUT=$(../cufinufft_type1 1 $DIM $N1 $N2 $N3 $M 1e-6)
acc=$(echo $OUT | grep -o -P '(?<=F is).*')
ns=$(echo $OUT | grep -o -P '(?<=ns=).*?(?= (\[time))')
echo 1 1 $N1 $N2 $N3 $M $ns $acc
OUT=$(../cufinufft_type2 1 $DIM $N1 $N2 $N3 $M 1e-6)
acc=$(echo $OUT | grep -o -P '(?<=c is).*')
ns=$(echo $OUT | grep -o -P '(?<=ns=).*?(?= (\[time))')
echo 1 2 $N1 $N2 $N3 $M $ns $acc

# CUNFFT
sh compile_cunfft_diff_tol.sh 1 >/dev/null 2>&1
OUT=$(../cunfft_type1 1 $DIM $N1 $N2 $N3 $M 1e-2)
acc=$(echo $OUT | grep -o -P '(?<=F is).*')
ns=$(echo $OUT | grep -o -P '(?<=ns=).*?(?= (\[time))')
echo 3 1 $N1 $N2 $N3 $M $ns $acc
OUT=$(../cunfft_type2 1 $DIM $N1 $N2 $N3 $M 1e-2)
acc=$(echo $OUT | grep -o -P '(?<=c is).*')
ns=$(echo $OUT | grep -o -P '(?<=ns=).*?(?= (\[time))')
echo 3 2 $N1 $N2 $N3 $M $ns $acc

sh compile_cunfft_diff_tol.sh 2 >/dev/null 2>&1
OUT=$(../cunfft_type1 1 $DIM $N1 $N2 $N3 $M 1e-4)
acc=$(echo $OUT | grep -o -P '(?<=F is).*')
ns=$(echo $OUT | grep -o -P '(?<=ns=).*?(?= (\[time))')
echo 3 1 $N1 $N2 $N3 $M $ns $acc
OUT=$(../cunfft_type2 1 $DIM $N1 $N2 $N3 $M 1e-4)
acc=$(echo $OUT | grep -o -P '(?<=c is).*')
ns=$(echo $OUT | grep -o -P '(?<=ns=).*?(?= (\[time))')
echo 3 2 $N1 $N2 $N3 $M $ns $acc

sh compile_cunfft_diff_tol.sh 3 >/dev/null 2>&1
OUT=$(../cunfft_type1 1 $DIM $N1 $N2 $N3 $M 1e-6)
acc=$(echo $OUT | grep -o -P '(?<=F is).*')
ns=$(echo $OUT | grep -o -P '(?<=ns=).*?(?= (\[time))')
echo 3 1 $N1 $N2 $N3 $M $ns $acc
OUT=$(../cunfft_type2 1 $DIM $N1 $N2 $N3 $M 1e-6)
acc=$(echo $OUT | grep -o -P '(?<=c is).*')
ns=$(echo $OUT | grep -o -P '(?<=ns=).*?(?= (\[time))')
echo 3 2 $N1 $N2 $N3 $M $ns $acc
