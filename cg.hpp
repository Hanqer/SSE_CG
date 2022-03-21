#pragma once
#ifndef CG
#include "utils.hpp"
#include <random>
#include <algorithm>

class CG_Solver
{
public:
	CG_Solver(int size) : size(size) {}
	~CG_Solver() {}
	void init_parameters();
	void init_Abx();
	mVector solve();
	friend int main();
private:
	mVector x, r, p, b;
	mMatrix A;
	float alpha, beta;
	int size;
};

mVector CG_Solver::solve()
{
	MyTimer timer;
	timer.start();
	int k_max = size;
	int k = 0;
	float eps = 1e-12;
	float rr = t_mul(r, r);
	while (1)
	{
		mVector ap = A * p;
		alpha = rr / t_mul(p, ap);

		x = x + p * alpha;
		r = r - ap * alpha;
		float rr_new = t_mul(r, r);

		if (rr_new < eps) break;

		beta = rr_new / rr;
		p = r + p * beta;

		rr = rr_new;
		k++;
		//std::cout << "INFO:" << k << "  " << rr_new << std::endl;
	}
	timer.stop();
	std::cout << "Time elapsed: " << timer.elapse() << std::endl;
	std::cout << "k =  " << k << std::endl;
	return x;
}

void CG_Solver::init_parameters()
{
	r = b - A*x;
	p = r;
}

void CG_Solver::init_Abx()
{
	//random x
	srand(1997);

	const int MAX = 50;
	const int MAX_M = 5;
	x = mVector(size);
	for (int i = 0; i < size; ++i) 
	{
		x.data[i] = rand() % MAX;
	}

	b = mVector(size);
	for (int i = 0; i < size; ++i)
	{
		b.data[i] = rand() % MAX;
	}
	A = mMatrix(size, size);
	for (int i = 0; i < size; ++i)
	{
		for (int j = 0; j < size; ++j)
		{
			A.data[i][j] = rand() % MAX_M;
		}
	}
	A = A * A.t();
}
#endif // !CG
