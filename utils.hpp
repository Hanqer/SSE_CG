#pragma once
#ifndef UTILS
#define UTILS
#include <assert.h>
#include <iostream>
#include <fstream>
#include <xmmintrin.h>  
#include <pmmintrin.h>
#include <time.h>
#include "pthread.h"
#pragma comment(lib,"pthreadVC2.lib")

class mVector;
class mMatrix;
class MyTimer;

class MyTimer
{
private:
	clock_t _freq;
	clock_t _start;
	clock_t _stop;
public:

	MyTimer()
	{
		_freq = CLOCKS_PER_SEC;
	}

	inline void start()
	{
		_start = clock();
	}

	inline void stop()
	{
		_stop = clock();
	}

	inline double elapse()
	{
		return 1e3 * double(_stop - _start) / _freq;
	}

	inline long long ticks()
	{
		return _stop - _start;
	}
};

class mVector
{
public:
	mVector() : data(NULL){}
	mVector(const int size) : size(size)
	{
		data = new float[size];
	}
	mVector(const mVector& m)
	{
		size = m.size;
		data = new float[size];
		memcpy(data, m.data, size * sizeof(float));
	}
	mVector(const float *m, int s_size)
	{
		size = s_size;
		data = new float[size];
		memcpy(data, m, size * sizeof(float));
	}
	~mVector()
	{
		delete data;
	}
public:
	float sum()
	{
		float s = 0.0f;
#ifndef PARAL
		for (int i = 0; i < size; ++i)
			s += data[i];
#endif // !PARAL
#ifdef PARAL
		int nloop = size / 4;
		float *result = new float[4];
		__m128 m, sum;
		sum = _mm_setzero_ps();
		for (int i = 0; i < nloop; ++i)
		{
			m = _mm_loadu_ps(data + 4 * i);
			sum = _mm_add_ps(sum, m);
		}
		sum = _mm_hadd_ps(sum, sum);
		sum = _mm_hadd_ps(sum, sum);
		_mm_storeu_ps(result, sum);
		s = result[0];
		delete result;
#endif // PARAL
		return s;
	}
	mVector& operator=(const mVector &m)
	{
		if (this != &m)
		{
			size = m.size;
			if (data != NULL)
			{
				delete data;
				data = NULL;
			}
			data = new float[size];
			memcpy(data, m.data, size * sizeof(float));
		}
		return *this;
	}

	mVector operator+(const mVector &other)
	{
		assert(size == other.size);
		mVector res(size);
#ifndef PARAL
		for (int i = 0; i < size; ++i)
		{
			res.data[i] = data[i] + other.data[i];
		}
#endif // !PARAL

#ifdef PARAL
		__m128 m1, m2, m3;
		float *p1 = data, *p2 = other.data, *p3 = res.data;
		int nloop = size / 4;
		for (int i = 0; i < nloop; ++i)
		{
			m1 = _mm_loadu_ps(p1);
			m2 = _mm_loadu_ps(p2);
			m3 = _mm_add_ps(m1, m2);
			_mm_storeu_ps(p3, m3);
			p1 += 4;
			p2 += 4;
			p3 += 4;
		}
#endif // PARAL
		return res;
	}

	mVector operator*(const float f)
	{
		mVector res(size);
#ifndef PARAL
		for (int i = 0; i < size; ++i)
		{
			res.data[i] = data[i] * f;
		}
#endif // !PARAL

#ifdef PARAL
		__m128 m1, m2, m3;
		float *p1 = data, *p3 = res.data;
		
		int nloop = size / 4;
		for (int i = 0; i < nloop; ++i)
		{
			m1 = _mm_loadu_ps(p1);
			m2 = _mm_set1_ps(f);
			m3 = _mm_mul_ps(m1, m2);
			_mm_storeu_ps(p3, m3);
			p1 += 4;
			p3 += 4;
		}
#endif // PARAL
		return res;
	}

	mVector operator*(const mVector &other)
	{
		assert(size == other.size);
		mVector res(size);
#ifndef PARAL
		for (int i = 0; i < size; ++i)
		{
			res.data[i] = data[i] * other.data[i];
		}
#endif // !PARAL
#ifdef PARAL
		__m128 m1, m2, m3;
		float *p1 = data, *p2 = other.data, *p3 = res.data;
		int nloop = size / 4;
		for (int i = 0; i < nloop; ++i)
		{
			m1 = _mm_loadu_ps(p1);
			m2 = _mm_loadu_ps(p2);
			m3 = _mm_mul_ps(m1, m2);
			_mm_storeu_ps(p3, m3);
			p1 += 4;
			p2 += 4;
			p3 += 4;
		}
#endif // PARAL
	
		return res;
	}

	mVector operator-(const mVector &other)
	{
		assert(size == other.size);
		mVector res(size);
#ifndef PARAL
		for (int i = 0; i < size; ++i)
		{
			res.data[i] = data[i] - other.data[i];
		}
#endif // !PARAL
#ifdef PARAL
		__m128 m1, m2, m3;
		float *p1 = data, *p2 = other.data, *p3 = res.data;
		int nloop = size / 4;
		for (int i = 0; i < nloop; ++i)
		{
			m1 = _mm_loadu_ps(p1);
			m2 = _mm_loadu_ps(p2);
			m3 = _mm_sub_ps(m1, m2);
			_mm_storeu_ps(p3, m3);
			p1 += 4;
			p2 += 4;
			p3 += 4;
		}
#endif // PARAL
	
		return res;
	}

	int size;
	float *data;
};

class mMatrix
{
public:
	mMatrix() : data(NULL) {}
	mMatrix(const int row, const int col) : row(row), col(col)
	{
		data = new float *[row];
		for (int i = 0; i < row; ++i)
		{
			data[i] = new float[col];
			memset(data[i], 0, col * sizeof(float));
		}
	}
	~mMatrix()
	{
		for (int i = 0; i < row; ++i)
		{
			delete data[i];
		}
		delete data;
	}
public:
	mMatrix & operator=(const mMatrix &m)
	{
		if (this != &m)
		{
			row = m.row;
			col = m.col;
			if (data != NULL)
			{
				for (int i = 0; i < row; ++i)
				{
					if (data[i] != NULL)
						delete data[i];
					data[i] = NULL;
				}
				delete data;
				data = NULL;
			}

			data = new float* [row];
			for (int i = 0; i < row; ++i)
			{
				data[i] = new float[col];
				memcpy(data[i], m.data[i], col * sizeof(float));
			}
		}
		return *this;
	}

	struct mul_thread_param
	{
		float *res;
		float **data;
		int size;
		int length;
		const mVector *b;
	};

	static void *mul_thread(void *arg)
	{
		
		mul_thread_param *param = (struct mul_thread_param *)arg;
		int length = param->length;
		for (int i = 0; i < length; ++i)
		{
			param->res[i] = (mVector(param->data[i], param->size) * (*(param->b))).sum();
		}
		
		return NULL;
	}

	mVector operator*(const mVector &other)
	{
		assert(other.size == col);
		assert(col == row);
		mVector res(row);
		/*
		MyTimer timer;
		timer.start();
		for (int i = 0; i < other.size; ++i)
		{
			res.data[i] = (mVector(data[i], col) * other).sum();
		}
		timer.stop();
		std::cout << "Time elapsed: " << timer.elapse() << std::endl;
		return res; 
		*/
		const int num_thread = 8;
		const int offset = other.size / num_thread;
		pthread_t *threads = new pthread_t[num_thread];
		mul_thread_param *arg = new mul_thread_param[num_thread];
		
		MyTimer timer2;
		timer2.start();
		for (int i = 0; i < num_thread; ++i)
		{
			arg[i].res = res.data + i*offset;
			arg[i].data = data + i*offset;
			arg[i].size = col;
			arg[i].b = &other;
			arg[i].length = offset;
			
			pthread_create(&threads[i], NULL, mul_thread, arg + i);
		}
		
		for (int i = 0, j = 0; i < other.size; i += offset, j++)
		{
			pthread_join(threads[j], NULL);
		}
		timer2.stop();
		std::cout << "Time elapsed(outer): " << timer2.elapse() << std::endl;
		delete threads;
		delete arg; 
		return res;
	}

	mMatrix operator*(const mMatrix &other)
	{
		assert(col == other.row);
		mMatrix res(row, other.col);
		for (int i = 0; i < row; ++i)
		{
			for (int j = 0; j < other.col; ++j)
			{
				for (int ii = 0; ii < col; ++ii)
				{
					res.data[i][j] += data[i][ii] * other.data[ii][j];
				}
			}
		}
		return res;
	}
	mMatrix t()
	{
		mMatrix t(row, col);
		for (int i = 0; i < row; ++i)
		{
			for (int j = 0; j < col; ++j)
			{
				t.data[j][i] = data[i][j];
			}
		}
		return t;
	}
	int row, col;
	float **data;
};

std::ostream & operator<<(std::ostream &os, const mMatrix &m)
{
	os << "[ ";
	for (int i = 0; i < m.row; ++i)
	{
		for (int j = 0; j < m.col; ++j)
		{
			os << m.data[i][j] << " ";
		}
		os << ";" << std::endl;
	}
	os << "]" << std::endl;
	return os;
}

std::ostream & operator<<(std::ostream &os, const mVector &m)
{
	os << "[ ";
	for (int i = 0; i < m.size; ++i)
	{
		os << m.data[i] << " ";
	}
	os << "]" << std::endl;
	return os;
}



float t_mul(mVector &l, mVector &r)
{
	mVector temp = l * r;
	return temp.sum();
}



#endif // !UTILS
