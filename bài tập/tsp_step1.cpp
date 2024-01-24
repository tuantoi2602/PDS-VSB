#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include <vector>


using namespace std;

vector<double> distances;

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

bool read_tsp_file(char * fname) 
{
	std::ifstream file(fname);
	
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

		unsigned int n = xs.size();

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

		print_distances(distances, n);
		file.close();
	}
	else 
	{
		cout << fname << " file not open" << endl;
	}
}

int main() 
{
	read_tsp_file("ulysses22.tsp.txt");
	return 0;
}
