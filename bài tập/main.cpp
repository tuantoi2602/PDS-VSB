#include <iostream>

#include <thread>
#include <future>
#include <chrono>
#include <random>

#include <vector>
#include <algorithm>
#include <utility>

using namespace std;

typedef vector<double> t_point;

/**
* Create a PRNG for every thread.
* TODO: make a safe initialization, use this_thread::get_id()
*/
double drand() {
	static thread_local mt19937 generator;
	uniform_int_distribution<unsigned int> distribution(0, 10);
	return distribution(generator);
}

/**
* Generate CNT random points of DIM dimensions. Single thread.
*/
vector<t_point> my_gen(unsigned int dim, unsigned int cnt)
{
	vector<t_point> results;

	t_point p;
	p.resize(dim);

	for (unsigned int i = 0; i < cnt; i++)
	{
		for (unsigned int j = 0; j < dim; j++)
			p[j] = drand();
		results.push_back(p);
	}
	return results;
}

/**
* Compute cumulative error (euclidean distance) between a centroid (m) and points in data in the range from FROM to TO.
*/
double my_err(vector<t_point> & data, t_point & m, unsigned int from, unsigned int to)
{
	double error = 0;
	double bit = 0;

	for (unsigned int i = from; i < fmin(to, data.size()); i++)
	{
		for (size_t j = 0; j < m.size(); j++)
		{
			bit += (m[j] - data[i][j])*(m[j] - data[i][j]);
		}
		error += sqrt(bit);
	}
	return error;
}

/**
* Helper function that just prints stuff.
*/
void print(vector<t_point> & data)
{
	for (t_point item : data)
	{
		for (double d : item)
			cout << d << "\t";
		cout << endl;
	}
}

/**
* Experimental function that:
*  1. generates CNT random DIM-dimensional points
*  2. selects <<randomly>> centroids
*  3. computes error of the clustering
*
* This all will be done in THREADS threads.
* Note 1: launch::async is not required for MSVC, but must be used for gcc, otherwise it will allways run in single thread.
*
*/
void experiment(const unsigned int DIM, const unsigned int CNT, const unsigned int THREADS)
{
	chrono::time_point<chrono::system_clock> start, end;
	start = chrono::system_clock::now();

	vector<future<vector<t_point>>> my_futures;

	for (unsigned int i = 0; i < THREADS; i++)
	{
		my_futures.push_back(async(launch::async, my_gen, DIM, CNT / THREADS));
	}

	vector<t_point> my_points;
	vector<t_point> current;

	for (unsigned int i = 0; i < my_futures.size(); i++)
	{
		current = my_futures[i].get();
		my_points.insert(my_points.end(), current.begin(), current.end());
	}

	end = chrono::system_clock::now();
	chrono::duration<double> elapsed_seconds = end - start;

	if (CNT < 40)
		print(my_points);

	cout << "Generating data...\nData " << CNT << ", Threads " << THREADS << ", Elapsed time " << elapsed_seconds.count() << "s\n";

	//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

	vector<future<double>> my_future_errs;

	start = chrono::system_clock::now();

	unsigned int piece = CNT / THREADS;
	for (unsigned int i = 0; i < THREADS; i++)
	{
		my_future_errs.push_back(async(launch::async, my_err, ref(my_points), ref(my_points[i]), i*piece, (i + 1)*piece));
	}

	double err = 0;

	for (unsigned int i = 0; i < my_future_errs.size(); i++)
	{
		err += my_future_errs[i].get();
	}

	end = chrono::system_clock::now();
	elapsed_seconds = end - start;

	cout << "Computing error...\nData " << CNT << ", Threads " << THREADS << ", Elapsed time " << elapsed_seconds.count() << "s\n" << " " << err << "\n";
}

int main()
{
	experiment(100, 1000000, 1);
	cout << endl;
	cout << endl;
	experiment(100, 1000000, 4);
}
