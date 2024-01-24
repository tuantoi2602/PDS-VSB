#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include <algorithm>
#include <vector>


using namespace std;

vector<double> distances;

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
double compute_cost(vector<double> & distances, vector<int> permutation, const unsigned int n_original)
{
	double cost = 0;
	for (unsigned int i = 0; i < permutation.size() - 1; i++) 
	{
		cost += distances[permutation[i] * n_original + permutation[i + 1]];
	}
	cost += distances[permutation[permutation.size() - 1] * n_original + permutation[0]];
	return cost;
}

// Tests
void test_tsp()
{
	const char * instance = "c:\\Users\\Pavel Krömer\\Documents\\Visual Studio 2015\\Projects\\cpp11-tsp\\Debug\\ulysses22.tsp.txt";
	unsigned int n = read_tsp_file(instance);
	if (n == 0) 
	{
		cout << "A problem occured" << endl;
	}
	else 
	{
		cout << "Loaded " << instance << " of dimension " << n << endl;
	}
}

void test_permutations() 
{
	unsigned int n = 5;
	vector<int> permutation;
	create_permutation(permutation, 5);
	print_permutation(permutation);
	while (next_permutation(permutation))
	{
		print_permutation(permutation);
	}
}

void test_permutations_prefix(unsigned int n, unsigned int p)
{
	vector<int> permutation;
	create_permutation_prefix(permutation, n, p);
	print_permutation(permutation);
	cout << endl;
	while (next_permutation(permutation, 2))
	{
		print_permutation(permutation);
		cout << endl;
	}
	cout << endl;
}

void test_permutation_tsp() 
{
	const char * instance = "c:\\Users\\Pavel Krömer\\Documents\\Visual Studio 2015\\Projects\\cpp11-tsp\\Debug\\ulysses22.tsp.txt";
	unsigned int n_orig = read_tsp_file(instance);
	const unsigned int n = 5; // We truncate to 5
	vector<int> permutation;
	create_permutation(permutation, n);
	print_permutation(permutation);
	cout << " " << compute_cost(distances, permutation, n_orig);
	cout << endl;
	while (next_permutation(permutation))
	{
		print_permutation(permutation);
		cout << " " << compute_cost(distances, permutation, n_orig);
		cout << endl;
	}
}

// Main
int main()
{
	//test_permutation_tsp();

	test_permutations_prefix(5, 1);
	test_permutations_prefix(5, 2);
	test_permutations_prefix(5, 3);
	
	return 0;
}
