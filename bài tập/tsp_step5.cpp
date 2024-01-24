#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>

#include <algorithm>
#include <vector>

#include <thread>
#include <future>

#include <chrono>

using namespace std;

vector<double> distances;

const double LARGE_VALUE = 100000000;
double cost_min = LARGE_VALUE;
mutex guard;

// Distances and TSP loading
void print_distances(vector<double> & dist, unsigned int n)
{
	for (unsigned int i = 0; i < dist.size(); i++)
	{
		cout << dist[i] << "\t";
		if ((i + 1) % n == 0)
			cout << endl;
	}
}

double compute_distance(double x1, double  y1, double  x2, double  y2)
{
	return sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));
}

unsigned int read_tsp_file(const char * fname)
{
	std::ifstream file(fname);

	unsigned int n = 0;

	if (file.is_open())
	{
		vector<double> xs, ys;

		std::string line;

		std::getline(file, line);
		std::getline(file, line);
		std::getline(file, line);
		std::getline(file, line);
		std::getline(file, line);
		std::getline(file, line);
		std::getline(file, line);

		while (std::getline(file, line)) {
			if (line[0] == 'E')
				break;

			stringstream sin(line);
			int id;
			double x, y;
			sin >> id >> x >> y;

			xs.push_back(x);
			ys.push_back(y);
		}

		n = xs.size();

		distances.resize(n*n);

		for (unsigned int i = 0; i < n; i++)
		{
			for (unsigned int j = i; j < n; j++)
			{
				double dist = compute_distance(xs[i], ys[i], xs[j], ys[j]);
				distances[i*n + j] = dist;
				distances[j*n + i] = dist;
			}
		}

		// print_distances(distances, n);
		file.close();
	}
	else
	{
		cout << fname << " file not open" << endl;		
	}
	return n;
}

// Permutations
void print_permutation(vector<int> & perm) 
{
	for (int i : perm) 
	{
		cout << i << " ";
	}
	//cout << endl;
}

void create_permutation(vector<int> & perm, unsigned int n)
{
	perm.clear();
	perm.resize(n);

	for (unsigned int i = 0; i < n; i++) 
	{
		perm[i] = i;
	}
}

void create_permutation_prefix(vector<int> & perm, unsigned int n, unsigned int p)
{
	perm.clear();
	perm.resize(n);

	for (unsigned int i = 0; i < n; i++)
	{
		perm[i] = i;
	}

	unsigned int r = perm[1];
	perm[1] = p;

	for (unsigned int i = 2; i < n; i++)
	{
		if (perm[i] == p) 
		{
			perm[i] = r;
			break;
		}		
	}

	std::sort(perm.begin() + 2, perm.end());
}

bool next_permutation(vector<int> & perm, unsigned int offset = 0)
{
	return std::next_permutation(perm.begin() + offset, perm.end());
}

// Compute tour cost
// Note: you can have a different solution here because I am storing the 
//       original distance matrix (n_orig = 22) and work only with the truncated data (n = 5).
double compute_cost(vector<double> & distances, vector<int> & permutation, const unsigned int n_original)
{
	double cost = 0;
	for (unsigned int i = 0; i < permutation.size() - 1; i++) 
	{
		cost += distances[permutation[i] * n_original + permutation[i + 1]];
	}
	cost += distances[permutation[permutation.size() - 1] * n_original + permutation[0]];
	return cost;
}

void tsp_thread(unsigned int p, double * result, unsigned int n, unsigned int n_original)
{
	vector<int> permutation;
	create_permutation_prefix(permutation, n, p);
	
	double best_tour = compute_cost(distances, permutation, n_original);
	double current_tour;	
	while (next_permutation(permutation, 2))
	{
		current_tour = compute_cost(distances, permutation, n_original);
		if (current_tour < best_tour) 
		{
			best_tour = current_tour;
		}
	}
	result[0] = best_tour;
}

double tsp_thread_ret(unsigned int p, unsigned int n, unsigned int n_original)
{
	vector<int> permutation;
	create_permutation_prefix(permutation, n, p);

	double best_tour = compute_cost(distances, permutation, n_original);
	double current_tour;
	while (next_permutation(permutation, 2))
	{
		current_tour = compute_cost(distances, permutation, n_original);
		if (current_tour < best_tour)
		{
			best_tour = current_tour;
		}
	}
	return best_tour;
}

void test_tsp_brute_force(const char * instance, unsigned int n_lim)
{
	unsigned int n = read_tsp_file(instance);
	if (n == 0)
	{
		cout << "A problem occured" << endl;
	}
	else
	{
		cout << "Loaded " << instance << " of dimension " << n << endl;
	}

	unsigned int n_original = n;
	n = n_lim; // we limit ourselves to 5 cities first

	vector<thread> threads;
	vector<double> results;
	results.resize(n - 1);

	for (unsigned int i = 1; i < n; ++i) {
		threads.push_back(
			thread(tsp_thread, i, &results[i - 1], n, n_original)
		);		
	}

	for (auto& thread : threads) {
		thread.join();
	}

	double min = results[0];
	for (unsigned int i = 1; i < results.size(); i++) 
	{
		if (results[i] < min)
			min = results[i];		
	}

	cout << "Minimum tour length is:" << min << endl;
}

void test_tsp_brute_force_async(const char * instance)
{
	unsigned int n = read_tsp_file(instance);
	if (n == 0)
	{
		cout << "A problem occured" << endl;
	}
	else
	{
		cout << "Loaded " << instance << " of dimension " << n << endl;
	}

	unsigned int n_original = n;
	n = 5; // we limit ourselves to 5 cities first

	vector<future<double>> results;

	for (unsigned int i = 1; i < n; ++i) {
		results.push_back(async(tsp_thread_ret, i, n, n_original));
	}

	double min_result = 100000;
	for (auto& result: results) {
		double current = result.get();
		if (current < min_result)
			min_result = current;
	}
	cout << "Minimum tour length is:" << min_result << endl;
}

double compute_cost_bnb(vector<double> & distances, vector<int> & permutation, const unsigned int n_original, unsigned int & backtrack)
{
	double cost = 0;
	backtrack = 0;
	for (unsigned int i = 0; i < permutation.size() - 1; i++)
	{
		cost += distances[permutation[i] * n_original + permutation[i + 1]];
		if (cost > cost_min)
		{
			backtrack = i;
			return -1;
		}
	}
	cost += distances[permutation[permutation.size() - 1] * n_original + permutation[0]];
	return cost;
}

void tsp_thread_bnb(unsigned int p, double * result, unsigned int n, unsigned int n_original)
{
	vector<int> permutation;
	create_permutation_prefix(permutation, n, p);
	unsigned int backtrack = 0;

	double best_tour = compute_cost_bnb(distances, permutation, n_original, backtrack);
	bool has_next = true;
	if (backtrack > 0)
	{
		best_tour = LARGE_VALUE;
		unsigned int bad_value = permutation[backtrack];
		do {
			has_next = next_permutation(permutation, 2);
		} while (has_next && (permutation[backtrack] == bad_value));		
	} 
	else
	{
		guard.lock();
		if (best_tour < cost_min)
		{
			cost_min = best_tour;
		}
		guard.unlock();
	}
	
	if (has_next == false)
	{
		result[0] = LARGE_VALUE;
		return;
	}
	
	double current_tour;
	while (next_permutation(permutation, 2))
	{
		current_tour = compute_cost_bnb(distances, permutation, n_original, backtrack);
		if (backtrack > 0)
		{
			current_tour = LARGE_VALUE;
			unsigned int bad_value = permutation[backtrack];
			bool has_next = true;
			do {
				has_next = next_permutation(permutation, 2);
			} while (has_next && (permutation[backtrack] == bad_value));
			
			if (!has_next)
				break;
		}

		if (current_tour < best_tour)
		{
			best_tour = current_tour;			
			guard.lock();
			if (best_tour < cost_min)
			{
				cost_min = best_tour;			
			}
			guard.unlock();
		}
	}
	result[0] = best_tour;
}

void test_tsp_bnb(const char * instance, unsigned int n_lim)
{
	unsigned int n = read_tsp_file(instance);
	if (n == 0)
	{
		cout << "A problem occured" << endl;
	}
	else
	{
		cout << "Loaded " << instance << " of dimension " << n << endl;
	}

	unsigned int n_original = n;
	n = n_lim; // we limit ourselves to 5 cities first

	vector<thread> threads;
	vector<double> results;
	results.resize(n - 1);

	for (unsigned int i = 1; i < n; ++i) {
		threads.push_back(
			thread(tsp_thread_bnb, i, &results[i - 1], n, n_original)
		);
	}

	for (auto& thread : threads) {
		thread.join();
	}

	double min = results[0];
	for (unsigned int i = 1; i < results.size(); i++)
	{
		if (results[i] < min)
			min = results[i];
	}

	cout << "Minimum tour length is:" << min << endl;
}

// Main
int main()
{	
	unsigned int n_lim = 8;
	auto start = std::chrono::system_clock::now();
	test_tsp_brute_force("c:\\Users\\Pavel Krömer\\Documents\\Visual Studio 2015\\Projects\\cpp11-tsp\\Debug\\ulysses22.tsp.txt", n_lim);
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;
	cout << "BF duration: " << std::setw(9) << diff.count() << endl;
	start = std::chrono::system_clock::now();
	
	// DISCLAIMER : There is still an error in the code, I need to debug it. But this is the basic template to follow.
	test_tsp_bnb("c:\\Users\\Pavel Krömer\\Documents\\Visual Studio 2015\\Projects\\cpp11-tsp\\Debug\\ulysses22.tsp.txt", n_lim);
	end = std::chrono::system_clock::now();
	diff = end - start;
	cout << "BnB duration:" << std::setw(9) << diff.count() << endl;
	
	return 0;
}
