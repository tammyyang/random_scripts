//g++ pla.cpp -o pla -std=c++0x

#include <iostream>
#include <numeric>
#include <vector>

using namespace std;

int main( void)
{
        vector< vector<int> > d;
        vector<int> w(3,0);

        // x,y, 1/-1
        d.push_back( { 1, 2, 1, 1} );
        d.push_back( { 1, 3, 1,-1} );
        d.push_back( { 1, 6, 1,-1} );
        d.push_back( { 1, 2, 2, 1} );
        d.push_back( { 1, 4, 2,-1} );
        d.push_back( { 1, 7, 2,-1} );
        d.push_back( { 1, 3, 3, 1} );
        d.push_back( { 1, 5, 3,-1} );
        d.push_back( { 1, 1, 4, 1} );
        d.push_back( { 1, 4, 4, 1} );
        d.push_back( { 1, 6, 4,-1} );

        cout << "w= ( " << w[0] << " , " << w[1] << " , " << w[2] << " )" << endl;
        for( auto i = d.begin(); i != d.end(); i++)
        {
                cout << "w= ( " << w[0] << " , " << w[1] << " , " << w[2] << " ) , Yn * Wt * Xn= " << (*i)[3]*inner_product( w.begin(), w.end(), i->begin(), 0) << endl;

                if( inner_product( w.begin(), w.end(), i->begin(), 0) * (*i)[3] > 0)
                        continue;

                do
                {
                        // w' = w + yx
                        w[0]= w[0] + (*i)[3]*(*i)[0];
                        w[1]= w[1] + (*i)[3]*(*i)[1];
                        w[2]= w[2] + (*i)[3]*(*i)[2];
                        cout << "w= ( " << w[0] << " , " << w[1] << " , " << w[2] << " ) , Yn * Wt * Xn= " << (*i)[3]*inner_product( w.begin(), w.end(), i->begin(), 0) << endl;
                        i= d.begin();
                } while( inner_product( w.begin(), w.end(), i->begin(), 0) * (*i)[3] <= 0);
        }

        return 0;
}
