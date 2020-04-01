#define PARAL

#include<iostream>
#include<stdlib.h>
#include "utils.hpp"
#include "cg.hpp"

using namespace std;

int main()
{
	
	CG_Solver cg(256);
	cg.init_Abx();
	cg.init_parameters();
	auto x = cg.solve();

	cout << x << endl;
	cout << "verify b = ";
	cout << cg.A * x - cg.b << endl;

	system("pause");
	return 0;
}