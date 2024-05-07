
prefix='ZnBL_x1geom'
# has the vac, a0, d, vdw, interc parameters
source ../vasp_input/struct.sh

make_subdir () {

	local supercell=$1
	local occstring=$2

        # id for filepath
	i="CELL${supercell}x${supercell}_OCC${occstring}"

	# create directories
	if [ ! -d "${prefix}_${i}" ]; then mkdir "${prefix}_${i}"; fi
	if [ ! -d "${prefix}_${i}/sp" ]; then mkdir "${prefix}_${i}/sp"; fi
	if [ ! -d "${prefix}_${i}/dos" ]; then mkdir "${prefix}_${i}/dos"; fi

	# copy in needed files
	if [ ! -f "${prefix}_${i}/sp/INCAR" ]; then
	  cp ../vasp_input/INCAR "${prefix}_${i}/sp"
	  cp ../vasp_input/POTCAR "${prefix}_${i}/sp"
	  cp ../vasp_input/KPOINTS "${prefix}_${i}/sp"
	  cp ../vasp_input/job.sh "${prefix}_${i}/sp"
	  cp ../vasp_input/INCAR_DOS "${prefix}_${i}/dos/INCAR"
	  cp ../vasp_input/POTCAR "${prefix}_${i}/dos"
	  cp ../vasp_input/KPOINTS "${prefix}_${i}/dos"
	  cp ../vasp_input/job.sh "${prefix}_${i}/dos"
	fi

	# write the poscar
	python3 make_supercell.py "$a0" "$d" "$vdw" "$vac" "${prefix}_${i}/sp/POSCAR" "$supercell" "$occstring" "$interc" > "${prefix}_${i}/supercell.log"
	cp "${prefix}_${i}/sp/POSCAR" "${prefix}_${i}/dos/POSCAR"
}

#make_subdir 1 '0'
#make_subdir 1 '1'
make_subdir 2 '1000'
#make_subdir 2 '1001'
#make_subdir 2 '1100'
#make_subdir 2 '1110'
#make_subdir 3 '100000000'

