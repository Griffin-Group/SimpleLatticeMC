

using SparseArrays, LinearAlgebra, Random, Statistics, Printf
   
# INIT OF COUPLING MATRIX FOR ENERGIES

function printLattice(N_IP::Int64, occupation::SparseVector{Int8, Int64}, iostream::IOStream)
	pad = "";
	for n in 0:N_IP-1
		for m in 0:N_IP-1
			index1 =  n*N_IP + m + 1;
			if occupation[index1] == 0
				write(iostream, ". ")
			else 
				write(iostream, "o ")
			end
		end 
		pad = pad * " ";
		write(iostream, "\n", pad)
	end
	write(iostream, "\n")
end

function printOP3(N_IP::Int64, occupation::SparseVector{Int8, Int64}, iostream::IOStream)
	pad = "";
	for n in 0:N_IP-1
		for m in 0:N_IP-1
			index1 =  n*N_IP + m + 1;
			if occupation[index1] == 0
				write(iostream, "  ")
			elseif (n%3 == 0 && m%3 == 0) || (n%3 == 1 && m%3 == 1) || (n%3 == 2 && m%3 == 2) 
				write(iostream, ". ")
			elseif (n%3 == 0 && m%3 == 1) || (n%3 == 1 && m%3 == 2) || (n%3 == 2 && m%3 == 0) 
				write(iostream, "o ")
			elseif (n%3 == 0 && m%3 == 2) || (n%3 == 1 && m%3 == 0) || (n%3 == 2 && m%3 == 1) 
				write(iostream, "+ ")
			end
		end 
		pad = pad * " ";
		write(iostream, "\n", pad)
	end
	write(iostream, "\n")
end

function printOP2(N_IP::Int64, occupation::SparseVector{Int8, Int64}, iostream::IOStream)
	pad = "";
	for n in 0:N_IP-1
		for m in 0:N_IP-1
			index1 =  n*N_IP + m + 1;
			if occupation[index1] == 0
				write(iostream,"  ")
			elseif n%2 == 0 && m%2 == 0
				write(iostream,". ")
			elseif n%2 == 0 && m%2 == 1
				write(iostream,"o ")
			elseif n%2 == 1 && m%2 == 0
				write(iostream,"x ")
			elseif n%2 == 1 && m%2 == 1
				write(iostream, "+ ")
			end
		end 
		pad = pad * " ";
		write(iostream,"\n", pad)
	end
	write(iostream,"\n")
end

function hinit( J1::Float64, J2::Float64, Jr3::Float64, N_IP::Int64 )

	rows::Vector{Int64} = zeros(0);
	cols::Vector{Int64} = zeros(0);
	vals::Vector{Float64} = zeros(0);

	index1::Int64 = 0;
	index2::Int64 = 0;
	idiff::Int64 = 0;
	jdiff::Int64 = 0;

	for i in 0:N_IP-1
		for j in 0:N_IP-1
			for ii in 0:N_IP-1
				for jj in 0:N_IP-1

					index1 =  i*N_IP + j;
					index2 = ii*N_IP + jj;

					# NO PBC YET!
					idiff = i-ii;
					jdiff = j-jj;

					# PBC 
					if idiff > N_IP/2
						idiff -= N_IP;
					elseif idiff < -N_IP/2
						idiff += N_IP;
					end 
					if jdiff > N_IP/2 
						jdiff -= N_IP;
					elseif jdiff < -N_IP/2
						jdiff += N_IP;
					end 
					
					# J1  between ( nm, n+1,m ) ( nm, n-1,m ) ( nm, n,m+1 ) ( nm, n,m-1 ) ( nm, n+1,m-1 ) ( nm, n-1,m+1 ) 
					if (abs(idiff)==1 && abs(jdiff)==0) || (abs(idiff)==0 && abs(jdiff)==1) || (idiff*jdiff==-1) 
						
						append!( rows, index1+1 );
						append!( cols, index2+1 );
						append!( vals, J1 );

					# Jr3 between ( nm, n+1,m+1 ) ( nm, n-1,m-1 ) ( nm, n-2,m+1 ) ( nm, n+2,m-1 ) ( nm, n+1,m-2 ) ( nm, n-1,m+2 ) 
					elseif (abs(idiff)==1 && idiff==jdiff) || (idiff*jdiff==-2) 

						append!( rows, index1+1 );
						append!( cols, index2+1 );
						append!( vals, Jr3 );

					# J2  between ( nm, n+2,m ) ( nm, n-2,m ) ( nm, n,m+2 ) ( nm, n,m-2 ) ( nm, n+2,m-2 ) ( nm, n-2,m+2 )
					elseif (abs(idiff)==2 && abs(jdiff)==0) || (abs(idiff)==0 && abs(jdiff)==2) || (abs(idiff)==2 && idiff*jdiff==-4) 

						append!( rows, index1+1 );
						append!( cols, index2+1 );
						append!( vals, J2 );

					end
	    		end 
			end
		end
	end 
	return sparse(rows, cols, vals, N_IP*N_IP, N_IP*N_IP)
end

function printham(N_IP::Int64, J1::Float64, J2::Float64, Jr3::Float64, ham::SparseMatrixCSC{Float64, Int64} )
	for i in 1:N_IP*N_IP
		for j in 1:N_IP*N_IP
			if ham[i,j] == 0
				print("   ")
			elseif ham[i,j] == J1
				print(" x ")
			elseif ham[i,j] == Jr3
				print(" o ")
			elseif ham[i,j] == J2
				print(" . ")
			end
			if ham[i,j] != ham[j,i]
				print(" NOT SYMMETRIC!")
			end
		end 
		print("\n")
	end 
end

function op3(N_IP::Int64, occupation::SparseVector{Int8, Int64})
	sl_occ::Vector{Int64} = [0,0,0]; # 3 different 2x2 sublattices
	index1::Int64 = 0;
	for n in 0:N_IP-1
		for m in 0:N_IP-1
			index1 =  n*N_IP + m;
			if (n%3 == 0 && m%3 == 0) || (n%3 == 1 && m%3 == 1) || (n%3 == 2 && m%3 == 2) 
				sl_occ[1] += occupation[index1+1];
			elseif (n%3 == 0 && m%3 == 1) || (n%3 == 1 && m%3 == 2) || (n%3 == 2 && m%3 == 0) 
				sl_occ[2] += occupation[index1+1];
			elseif (n%3 == 0 && m%3 == 2) || (n%3 == 1 && m%3 == 0) || (n%3 == 2 && m%3 == 1) 
				sl_occ[3] += occupation[index1+1];
			end
		end 
	end
	#print(sl_occ)
	return (maximum(sl_occ) - minimum(sl_occ))/(N_IP*N_IP)*3
end

function op2(N_IP::Int64, occupation::SparseVector{Int8, Int64})
	sl_occ::Vector{Int64} = [0,0,0,0]; # 4 different 2x2 sublattices
	index1::Int64 = 0;
	for n in 0:N_IP-1
		for m in 0:N_IP-1
			index1 =  n*N_IP + m;
			if n%2 == 0 && m%2 == 0
				sl_occ[1] += occupation[index1+1];
			elseif n%2 == 0 && m%2 == 1
				sl_occ[2] += occupation[index1+1];
			elseif n%2 == 1 && m%2 == 0
				sl_occ[3] += occupation[index1+1];
			elseif n%2 == 1 && m%2 == 1
				sl_occ[4] += occupation[index1+1];
			end
		end 
	end
	#print(sl_occ)
	return (maximum(sl_occ) - minimum(sl_occ))/(N_IP*N_IP)*4
end 

function mcmain( J1::Float64, J2::Float64, Jr3::Float64, N_IP::Int64, kbT::Float64, x::Float64, sweeps::Int64, warmup_sweeps::Int64, iostream::IOStream, sumpath::String, uselog::Bool)

	println(iostream, "##############################################################################")
	println(iostream, "beginging with x=", x, " kbt =", kbT, "meV, in-plane N of", N_IP)
	println(iostream, "and coupling constants of J1, Jr3, J2 = ", J1, "meV, ", Jr3, "meV, ", J2, "meV")
	println(iostream, "##############################################################################")

	N_full::Int64  = trunc(Int64, round(x*N_IP*N_IP));
	println(iostream, "filling ", N_full, " of ", N_IP*N_IP, " sites for x=", N_full/(N_IP*N_IP));

	coupling::SparseMatrixCSC{Float64, Int64} = hinit( J1, J2, Jr3, N_IP );
	#printham(N_IP, J1, J2, Jr3, coupling)
	onetemp::Vector{Int8} = ones(N_full);
	permtemp::Vector{Int64} = randperm(N_IP*N_IP);
	occupation::SparseVector{Int8, Int64} = sparsevec(permtemp[1:N_full], onetemp, N_IP*N_IP);
	hop_directions::Vector{Vector{Int8}} = [ [0,-1], [0,1], [1,0], [-1,0], [1,-1], [-1,1] ];

	beta::Float64 = 1/kbT;
	hsigma::SparseVector{Float64, Int64} = coupling * occupation;
	energy::Float64 = occupation' * hsigma;

	# allocations - trying to reuse as much as possible 
	N_accept::Int64   = 0;
	N_reject::Int64   = 0;
	N_fail::Int64     = 0;
	steps::Int64      = 0;
	siteindx1::Int64  = 0;
	siteindx2::Int64  = 0;
	dE::Float64 = 0;
	m::Int64    = 0;
	k::Int64    = 0;
	hop_dir::Vector{Int8} = [0,0];
	ii::Int64    = 0;
	jj::Int64    = 0;
	i::Int64    = 0;
	j::Int64    = 0;
	N2::Int64   = (N_IP * N_IP);
	o1::Int8 = 0;
	o2::Int8 = 0;
	f1::Float64 = 0;
	f2::Float64 = 0;
	f3::Float64 = 0;
	tempv::SparseVector{Float64, Int64} = coupling[:,1];

	for sweep in 0:warmup_sweeps 

		steps = 0

		while steps < N2

			# will use kawasaki dynamics : correct detailed balance if select direction
			# and sites randomly and fail if incompatible for a hop update, note that 
			# if only pick from "allowed" site pairs for the hop detailed balance breaks.
			# pick random site
			siteindx1 = rand(1:N2) # siteindx1 - 1 = N_IP*i + j
			i = (siteindx1 - 1)÷N_IP # floordiv to pull i from compund
			j = (siteindx1 - 1)%N_IP # mod to pull from compund 

			# pick random direction to hop
			hop_dir = hop_directions[rand(1:6)]
			ii = (i+hop_dir[1])%N_IP # mod to wrap around pbc
			jj = (j+hop_dir[2])%N_IP # mod to wrap around pbc
			if ii < 0
				ii += N_IP # more pbc handing
			end 
			if jj < 0 
				jj += N_IP
			end

			siteindx2 = N_IP*ii + jj + 1
			o1 = occupation[siteindx1];
			o2 = occupation[siteindx2];

			if o1 == o2
				N_fail += 1;
				continue
			elseif  o1 == 1 && o2 == 0
				m = siteindx1
				k = siteindx2
				steps += 1;
			else
				m = siteindx2
				k = siteindx1
				steps += 1;
			end
			
			# energy change
			dE = coupling[m,m] + coupling[k,k] - 2*coupling[m,k] + 2*hsigma[k] - 2*hsigma[m]; # little expensive 
			f1 = rand();
			f2 = -beta*dE;
			f3 = exp(f2);

			# do the metropolis hastings - default rand() is uniform in 0,1
			if (f1 <= f3) 
				occupation[m] = 0; # prev occ site 		    # little expensive 
				occupation[k] = 1; # hop destination 		# little expensive 
				# NOTE: symmetric, way faster to use choose column major slice
				tempv = coupling[:,k] - coupling[:,m];      # BOTTLENECK pt 1 - less gc w alloced temp 
				hsigma = hsigma + tempv; 					# BOTTLENECK pt 2, note .+, .- slower. vec worse?
				energy += dE; 
				N_accept += 1;
				#print("hs ---->", round(hsigma[4], digits=4) - round((coupling * occupation)[4], digits=4), "\n")
				#print("E ---->", round(energy, digits=4) - round(occupation' * coupling * occupation, digits=4), "\n")
			else
				N_reject += 1;
			end
		end 
	end 

	println(iostream, "##############################################################################")
	println(iostream, "after ", warmup_sweeps, " warmup sweeps ", warmup_sweeps*N_IP*N_IP, " steps")
	println(iostream, "percent accepted ", 100*N_accept/(N_accept+N_reject+N_fail))
	println(iostream, "percent rejected ", 100*N_reject/(N_accept+N_reject+N_fail))
	println(iostream, "percent failed to hop ", 100*N_fail/(N_accept+N_reject+N_fail))
	println(iostream, "##############################################################################")

	N_accept = 0; # already allocated
	N_reject = 0;
	N_fail   = 0;

	# allocations, rest resued from warmup loop
	op2_x::Vector{Float64}   = zeros( trunc(Int64, sweeps ÷ 5) + 1 ); 
	op3_x::Vector{Float64}   = zeros( trunc(Int64, sweeps ÷ 5) + 1 );

	printLattice(N_IP, occupation, iostream)
	println(iostream, "##############################################################################")

	for sweep in 0:sweeps 

		steps = 0

		while steps < N2

			# will use kawasaki dynamics : correct detailed balance if select direction
			# and sites randomly and fail if incompatible for a hop update, note that 
			# if only pick from "allowed" site pairs for the hop detailed balance breaks.
			# pick random site
			siteindx1 = rand(1:N2) # siteindx1 - 1 = N_IP*i + j
			i = (siteindx1 - 1)÷N_IP # floordiv to pull i from compund
			j = (siteindx1 - 1)%N_IP # mod to pull from compund 

			# pick random direction to hop
			hop_dir = hop_directions[rand(1:6)]
			ii = (i+hop_dir[1])%N_IP # mod to wrap around pbc
			jj = (j+hop_dir[2])%N_IP # mod to wrap around pbc
			if ii < 0
				ii += N_IP # more pbc handing
			end 
			if jj < 0 
				jj += N_IP
			end

			siteindx2 = N_IP*ii + jj + 1
			o1 = occupation[siteindx1];
			o2 = occupation[siteindx2];

			if o1 == o2
				N_fail += 1;
				continue
			elseif  o1 == 1 && o2 == 0
				m = siteindx1
				k = siteindx2
				steps += 1;
			else
				m = siteindx2
				k = siteindx1
				steps += 1;
			end
			
			# energy change
			dE = coupling[m,m] + coupling[k,k] - 2*coupling[m,k] + 2*hsigma[k] - 2*hsigma[m]; # little expensive 
			f1 = rand();
			f2 = -beta*dE;
			f3 = exp(f2);

			# do the metropolis hastings - default rand() is uniform in 0,1
			if (f1 <= f3) 
				occupation[m] = 0; # prev occ site 		    # little expensive 
				occupation[k] = 1; # hop destination 		# little expensive 
				tempv = coupling[:,k] - coupling[:,m];
				hsigma = hsigma + tempv; # TRUE BOTTLENECK
				energy += dE; 
				N_accept += 1;
				#print("hs ---->", round(hsigma[4], digits=4) - round((coupling * occupation)[4], digits=4), "\n")
				#print("E ---->", round(energy, digits=4) - round(occupation' * coupling * occupation, digits=4), "\n")
			else
				N_reject += 1;
			end
		end 

		if sweep % 5 == 0
			op2_x[ 1 + trunc(Int64, sweep ÷ 5) ]  = op2(N_IP, occupation);
			#print(op2(N_IP, occupation), "   ", 1 + trunc(Int64, sweep ÷ 5), "\n");
			op3_x[ 1 + trunc(Int64, sweep ÷ 5) ]  = op3(N_IP, occupation);
		end

		if sweep % (sweeps/20) == 0
			i = trunc(Int64, sweep ÷ (sweeps/20));
			println(iostream, " --> <OP2> ", round(mean(op2_x), digits=4), " <OP3> ", round(mean(op3_x), digits=4), " at snapshot ", i, "/20")
		end 

	end 

	println(iostream, "##############################################################################")
	printLattice(N_IP, occupation, iostream)
	println(iostream, "3x3 OP: \n")
	printOP3(N_IP, occupation, iostream)
	println(iostream, "2x2 OP: \n")
	printOP2(N_IP, occupation, iostream)

	mop2::Float64  = mean(op2_x);
	s2op2::Float64 = mean((op2_x .- mop2).^2);
	s4op2::Float64 = mean((op2_x .- mop2).^4);
	bop2::Float64  = 1 - (s4op2/(3*s2op2*s2op2));
	mop3::Float64  = mean(op3_x);
	s2op3::Float64 = mean((op3_x .- mop3).^2);
	s4op3::Float64 = mean((op3_x .- mop3).^4);
	bop3::Float64  = 1 - (s4op3/(3*s2op3*s2op3));

	println(iostream, "##############################################################################")
	println(iostream, "after ", warmup_sweeps, " sweeps ", warmup_sweeps*N_IP*N_IP, " steps")
	println(iostream, " <OP2> ", mop2)
	println(iostream, " Variance OP2 ", s2op2 )
	println(iostream, " Binder Cum of OP2 ", bop2)
	println(iostream, " <OP3> ", mop3)
	println(iostream, " Variance OP3 ", s2op3 )
	println(iostream, " Binder Cum of OP3 ", bop3)
	println(iostream, "percent accepted ", 100*N_accept/(N_accept+N_reject+N_fail))
	println(iostream, "percent rejected ", 100*N_reject/(N_accept+N_reject+N_fail))
	println(iostream, "percent failed to hop ", 100*N_fail/(N_accept+N_reject+N_fail))
	println(iostream, "##############################################################################")

	if uselog
		sumio = open(sumpath, "a") 
		str = @sprintf("%4.5f \t %4.5f \t %4.5f \t %4.5f \t %4.5f \t %4.5f \t %4.5f \t %4.5f", x, kbT, mop2, s2op2, bop2, mop3, s2op3, bop3);
		println(sumio, str)
		close(sumio);
	else 
		str = @sprintf("%4.5f \t %4.5f \t %4.5f \t %4.5f \t %4.5f \t %4.5f \t %4.5f \t %4.5f \n", x, kbT, mop2, s2op2, bop2, mop3, s2op3, bop3);
		print(str)
	end 
end

function serial_main_ZnBL()

	# Zn bilayer
	J1::Float64 = 205.8; # all coupling in meV
	J2::Float64 = 5.9;
	Jr3::Float64  = 19.1;
	N_IP::Int64  = 18; # in plane dimension, 6 makes a 6x6 2d lattice
	sweeps::Int64 = 1e2;# 5e4;
	warmup_sweeps::Int64 = 1e2; 

	savedir::String = @sprintf("ZnBL_testspeed");
	#savedir::String = @sprintf("ZnBL_%4.1f", N_IP*N_IP);
	mkpath( savedir );
	sumpath::String = savedir * "/" * "mc_summary.txt";
	sumio::IOStream = open(sumpath, "a");
	println(sumio, "x \t\t kbt \t\t <op2> \t\t var(op2) \t binder(op2) \t\t <op3> \t\t var(op3) \t binder(op3) ");
	close(sumio);

	xs::Vector{Float64} = [0.10];
	# 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 
	# 0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40, 0.10, 0.11, 0.12, 0.13, 
	# 0.14, 0.15, 0.16, 0.17, 0.18, 0.19,

	kbTs::Vector{Float64} = [25];
	# 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 
	# 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65

	for x in xs
		for kbT in kbTs
			path::String = savedir * "/" * @sprintf("mc_x_%5.3f_kbT_%5.3f.txt", x, kbT)
			print("Will be saving to ", path, "\n")
			ios::IOStream = open(path, "a");
			mcmain(J1, J2, Jr3, N_IP, kbT, x, sweeps, warmup_sweeps, ios, sumpath, true)
			close(ios);
		end
	end
end

function multithread_main_ZnBL()

	# Zn bilayer
	J1::Float64 = 205.8; # all coupling in meV
	J2::Float64 = 5.9;
	Jr3::Float64  = 19.1;
	N_IP::Int64  = 12; # in plane dimension, 6 makes a 6x6 2d lattice
	sweeps::Int64 = 5e4;
	warmup_sweeps::Int64 = 5e4; 

	#savedir::String = @sprintf("ZnBL_testspeed");
	savedir::String = @sprintf("ZnBL_%4.1f", N_IP*N_IP);
	mkpath( savedir );
	print("Will be using ", Threads.nthreads(), " thread \n")
	print("x \t\t kbt \t\t <op2> \t\t var(op2) \t binder(op2) \t <op3> \t\t var(op3) \t binder(op3) \n");	

	xs::Vector{Float64} = [ 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 
							0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29,
							0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 
							0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50]

	kbTs::Vector{Float64} = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 
												 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 
												 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 
												 60, 61, 62, 63, 64, 65];

	Threads.@threads for x in xs
		for kbT in kbTs
			path::String = savedir * "/" * @sprintf("mc_x_%5.3f_kbT_%5.3f.txt", x, kbT)
			#print("Will be saving to ", path, "\n")
			ios::IOStream = open(path, "a");
			mcmain(J1, J2, Jr3, N_IP, kbT, x, sweeps, warmup_sweeps, ios, "", false)
			close(ios);
		end
	end
end

function multithread_main_ScBL()

	# Zn bilayer
	J1::Float64 = 281.0; # all coupling in meV
	J2::Float64 = 6.0;
	Jr3::Float64  = 19.1;
	N_IP::Int64  = 12; # in plane dimension, 6 makes a 6x6 2d lattice
	sweeps::Int64 = 5e4;
	warmup_sweeps::Int64 = 5e4; 

	#savedir::String = @sprintf("ZnBL_testspeed");
	savedir::String = @sprintf("ScBL_%4.1f", N_IP*N_IP);
	mkpath( savedir );
	print("Will be using ", Threads.nthreads(), " thread \n")
	print("x \t\t kbt \t\t <op2> \t\t var(op2) \t binder(op2) \t <op3> \t\t var(op3) \t binder(op3) \n");	

	xs::Vector{Float64} = [ 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 
							0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29,
							0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 
							0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50]

	kbTs::Vector{Float64} = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 
												 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 
												 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 
												 60, 61, 62, 63, 64, 65];

	Threads.@threads for x in xs
		for kbT in kbTs
			path::String = savedir * "/" * @sprintf("mc_x_%5.3f_kbT_%5.3f.txt", x, kbT)
			#print("Will be saving to ", path, "\n")
			ios::IOStream = open(path, "a");
			mcmain(J1, J2, Jr3, N_IP, kbT, x, sweeps, warmup_sweeps, ios, "", false)
			close(ios);
		end
	end
end

@time multithread_main_ScBL();
#@time serial_main_ZnBL();

