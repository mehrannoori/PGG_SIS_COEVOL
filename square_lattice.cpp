#include <iostream>
#include <cstdlib>
#include <random>
#include <vector>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>
#include <algorithm>

using namespace std;
using namespace std::chrono;

/*************************************************************************/
/*                        Parameters and variables                       */
/*************************************************************************/
const int D    =   100;       // row ond col of lattice
const int N    =   D*D;       // number of nodes

int   I0       =   N *0.01;  // initial infected node
int   C0       =   N *  0.5;
float Mu       =   0.3;       // recovery probability
float alpha_0  =   0.25;      // infection probability 0.7
float alpha_t  =   0.01;      // altruism coefficient
float alpha_r  =   0.01;      // self-interest coefficient

float g        =      5;       // number of groups members
float K_noise  =    0.5;       // noise
float r        =      3;       // enhancement factor
float C_i[]    = {0,10};      // cost of infected node 5
float C_g[]    =  {0,1};       // cost of cooperation
float tau      =   0.01;

int   mcs_max  =  10*N;      // montecarlo steps
int   ens_max  =      1;
int   l_twin   =  10000;
int   idx_twin =  mcs_max - l_twin;
int   N_step   =  N;

vector<vector<int>> L(N, vector<int>(5, 0));
vector<vector<vector<int>>> G(N, vector<vector<int>>(5, vector<int>(5, 0)));

vector<int> state(N, 0);
vector<int> strategy(N, -1);

string ENSEMBLE;
string IDENTIFIER;

random_device rd;
default_random_engine gen(rd());                 // random number generator
uniform_int_distribution<int> rand_node(0, N-1);        //random_node(gen)
uniform_real_distribution<double> rand_num(0.0, 1.0); //random_num(gen)
//uniform_int_distribution<int> rand_state(0,1);

/*************************************************************************/





/*************************************************************************/
/*                          Function Declaration                         */
/*************************************************************************/
void   init_state();
void   MCS();
void   PGG(int);
void   SIS(int);
float  set_payoff(int);
bool   W(float, float);
void   make_network();
void   set_neighbors();
void   set_groups();

void   time_evolution();
void   X_evolution(string, float&, double, double, double);
void   X_Y_evolution(string, string,
                   float&, double, double, double,
                   float&, double, double, double);
float* count();
void   print_parameters(ofstream&);
/*************************************************************************/




/*************************************************************************/
/*                                  Main                                 */
/*************************************************************************/
int main(int argc, char* argv[]){
    try{
        IDENTIFIER = string(argv[1]);
        ENSEMBLE = string(argv[2]);
    }
    catch(...){
        std::cerr << "No args is given! \nexample: ./a.out 1 1" << endl;
        return 0;
    }

    auto time_begin = high_resolution_clock::now();

    make_network();
    
    //init_state();
    time_evolution();
    // X_evolution("r", r, 1, 10, 0.1);
    //X_evolution("a0", alpha_0, 0, 1, 0.01);
    //X_evolution("ar", alpha_r, 0, 0.5, 0.005);
    //X_evolution("tau", tau, 0, 0.5, 0.005);
    //X_evolution("at", alpha_t, 0, 0.5, 0.005);
    //X_Y_evolution("r", "a0", r, 1, 10, 0.1, alpha_0, 0, 1, 0.01);
   

    auto time_end = high_resolution_clock::now();

    cout << endl 
         << "Finish.\nDuration: "
         << duration_cast<minutes>(time_end - time_begin).count() << " min."
         << duration_cast<hours>(time_end - time_begin).count() << " H." << endl;

    return 0;
}
/*************************************************************************/










/*************************************************************************/
/*                               Dynamic                                 */
/*************************************************************************/
void time_evolution(){
    init_state();
    string filename = "t-evol-" + IDENTIFIER + "-ens-" + ENSEMBLE + ".txt"; 
    ofstream output(filename);
    print_parameters(output);

    // montecarlo steps
    for(int mcs=1; mcs<mcs_max+1; mcs++){
        cout << mcs << endl;
        float* stst = count();
        output << mcs << ',' 
               << stst[0] << ',' << stst[1] << ',' << stst[2] << ','
               << stst[3] << ',' << stst[4] << ',' << stst[5] << endl;
        MCS();
    }
    output.close();
}


void X_evolution(string X_name, float& X, double X_begin, double X_end, double dx){
    X = X_begin;
    ofstream output(X_name + "-evol-" + IDENTIFIER + "-ens-" + ENSEMBLE + ".txt");
    print_parameters(output);

    while (X <= X_end){
        cout << X << endl; // cout for log file
        init_state();
        float avr_density[6] = {0,0,0,0,0,0};
        vector<vector<float>> DENSITY(6, vector<float>(mcs_max, 0));

        for(int mcs=0; mcs<mcs_max; mcs++){
            float* stst = count();
            for(int type=0; type<6; type++){
                DENSITY[type][mcs] = stst[type];
            }
            MCS();
        }

        for(int mcs=idx_twin; mcs<mcs_max; mcs++){
            for(int type=0; type<6; type++){
                avr_density[type] += DENSITY[type][mcs];
            }
        }

        for(int type=0; type<6; type++){
            avr_density[type] = avr_density[type] / (float)l_twin;
        }

        output << X << ',' 
               << avr_density[0] << ',' << avr_density[1] << ',' << avr_density[2] << ','
               << avr_density[3] << ',' << avr_density[4] << ',' << avr_density[5] << endl;

        X += dx;
    }
}


void X_Y_evolution(string X_name, string Y_name,
                   float& X, double X_begin, double X_end, double dx,
                   float& Y, double Y_begin, double Y_end, double dy)
{
    Y = Y_begin; // Y-axis
    ofstream output(X_name + '-' + Y_name + "-evol-" + 
                    IDENTIFIER + "-ens-" + ENSEMBLE + ".txt");

    print_parameters(output);

    while(Y <= Y_end){

        cout << Y << endl;
        X = X_begin; // X-axis

        while(X <= X_end){
        
            init_state();
            float avr_density[6] = {0,0,0,0,0,0};
            vector<vector<float>> DENSITY(6, vector<float>(mcs_max, 0));

            for(int mcs=0; mcs<mcs_max; mcs++){
                float* stst = count();
                for(int type=0; type<6; type++){
                    DENSITY[type][mcs] = stst[type];
                }
                MCS();
            }

            for(int mcs=idx_twin; mcs<mcs_max; mcs++){
                for(int type=0; type<6; type++){
                    avr_density[type] += DENSITY[type][mcs];
                }
            }

            for(int type=0; type<6; type++){
                avr_density[type] = avr_density[type] / (float)l_twin;
            }

            output << avr_density[0] << ',' << avr_density[1] << ',' << avr_density[2] << ','
               << avr_density[3] << ',' << avr_density[4] << ',' << avr_density[5] << ";";

            X += dx;
        }
        output << endl;
        Y += dy;
    }
    output.close();
}


float* count(){
    int stst_num[6] = {0,0,0,0,0,0};
    static float stst_density[6] = {0,0,0,0,0,0};
    for(int i=0; i<N; ++i){
        stst_num[0] +=  strategy[i];              //P_c
        stst_num[1] +=  state   [i];              //P_i
        stst_num[2] +=  strategy[i] &&  state[i]; //IC
        stst_num[3] += !strategy[i] &&  state[i]; //ID
        stst_num[4] +=  strategy[i] && !state[i]; //SC
        stst_num[5] += !strategy[i] && !state[i]; //SD
    }
    for(int i=0; i<6; ++i){
        stst_density[i] = stst_num[i] / (float)N;
    }
    return stst_density;
}
/*************************************************************************/









/*************************************************************************/
/*                       Model Function Definition                       */
/*************************************************************************/
void MCS(){
    //N_step elementary step
    for(int elem = 0; elem < N_step; elem++){
        int X = rand_node(gen);

        if( rand_num(gen) < tau){
            PGG(X);
        }
        else{
            SIS(X);
        }
    }
}



void SIS(int X){
    // ----- recovery ----- //
    if(state[X]==1){
        if( rand_num(gen) < Mu){
            state[X] = 0;
        }
    }

    else{
        for(int j=1; j<5; j++){
            if(state[L[X][j]] == 1){

                if(strategy[X]==0){

                    // SD + ID ----a0 ----> ID + ID
                    if(strategy[L[X][j]] == 0){
                        if( rand_num(gen) < alpha_0 ){ state[X] = 1; break; }
                    }

                    // SD + IC ----a0.at----> ID + IC
                    if(strategy[L[X][j]] == 1){
                        if( rand_num(gen) < (alpha_0*alpha_t) ){ state[X] = 1; break; }
                    }
                }

                if(strategy[X]==1){

                    // SC + IC ---a0.ar.at---> IC + IC
                    if(strategy[L[X][j]] == 1){
                        if( rand_num(gen) < (alpha_0*alpha_t*alpha_r) ){ state[X] = 1; break; }
                    }

                    // SC + ID ---a0.ar---> IC + ID
                    if(strategy[L[X][j]] == 0){
                        if( rand_num(gen) < (alpha_0*alpha_r) ){ state[X] = 1; break; }
                    }
                }
            }
        }
    }
}



void PGG(int X){
    // X -> Ok
    int Y = L[X][(rand()%(5-1))+1];

    float P_x = set_payoff(X);
    float P_y = set_payoff(Y);

    if(W(P_x, P_y)==true){ strategy[X] = strategy[Y]; }
}



float set_payoff(int x){

    float P_x = 0;

    for(int n=0; n < G[x].size(); n++){

        int Nc = 0;
        for(int m=0; m < G[x][n].size(); m++){
            Nc += strategy[G[x][n][m]];
        }

        P_x += ((r*Nc*C_g[1]) / (float) g) - C_g[strategy[x]] - C_i[state[x]];
    }

    return P_x;
}



bool W(float P_x, float P_y){
    if(rand_num(gen) < (1.0 / ( 1.0 + exp( (P_x-P_y) / K_noise) ))){
          return true;  }
    else{ return false; }
}



void init_state(){
    for(int i=0; i<C0; ++i){
        strategy[i] = 1; // Cooperator=1, Defector=0 
    }
    for(int i=C0; i<N; ++i){
        strategy[i] = 0;
    }
    std::shuffle(strategy.begin(), strategy.end(), default_random_engine(rd()));

    for(int i=0; i<N; i++){
        state[i] = 0;
    }

    int rd = 0;           // Susceptibles=0, infected=1
    for(int i=0; i<I0; i++){
        rd = rand_node(gen);
        state[rd] = 1;
    }
}

/*************************************************************************/






/*************************************************************************/
/*                                Network                                */
/*************************************************************************/
void neighbor(int i, int j){
    int i_prev = i-1;
    int i_next = i+1;

    int j_prev = j-1;
    int j_next = j+1;

    if(i==0){i_prev = (D-1) - i;}
    if(i==(D-1)){i_next = (D-1) - i;}

    if(j==0){j_prev = (D-1) - j;}
    if(j==(D-1)){j_next = (D-1) - j;}

    L[i*D+j][0] = i*D + j;
    L[i*D+j][1] = i_prev*D + j;
    L[i*D+j][2] = i*D + j_prev;
    L[i*D+j][3] = i_next*D + j;
    L[i*D+j][4] = i*D + j_next;
}


void set_neighbors(){
    for(int i=0; i<D; i++){
        for(int j=0; j<D; j++){
            neighbor(i, j);
        }
    }
}


void set_groups(){
    for(int i=0; i<G.size(); i++){
        for(int n=0; n<G[i].size(); n++){
            for(int m=0; m<G[i][n].size(); m++){
                G[i][n][m] = L[L[i][n]][m];
            }
        }
    }
}


void make_network(){
    set_neighbors();
    set_groups();
}
/*************************************************************************/





/*************************************************************************/
void print_parameters(ofstream& output){
    output << "#--------------------------------------#" << endl
           << "#" << setw(15) << right << "N:"         << setw(10) << N        << endl 
           << "#" << setw(15) << right << "I0:"        << setw(10) << I0       << endl
           << "#" << setw(15) << right << "C0:"        << setw(10) << C0       << endl
           << "#" << setw(15) << right << "neighbors:" << setw(10) << g        << endl
           << "#" << setw(15) << right << "gorups:"    << setw(10) << g        << endl
           << "#" << setw(15) << right << "Mu:"        << setw(10) << Mu       << endl
           << "#" << setw(15) << right << "alpha_0:"   << setw(10) << alpha_0  << endl
           << "#" << setw(15) << right << "alpha_t:"   << setw(10) << alpha_t  << endl
           << "#" << setw(15) << right << "alpha_r:"   << setw(10) << alpha_r  << endl
           << "#" << setw(15) << right << "r:"         << setw(10) << r        << endl 
           << "#" << setw(15) << right << "k:"         << setw(10) << K_noise  << endl 
           << "#" << setw(15) << right << "Ci:"        << setw(10) << C_i[1]   << endl
           << "#" << setw(15) << right << "Cg:"        << setw(10) << C_g[1]   << endl
           << "#" << setw(15) << right << "mcs_max:"   << setw(10) << mcs_max  << endl
           << "#" << setw(15) << right << "ens_max:"   << setw(10) << ens_max  << endl
           << "#" << setw(15) << right << "l_twin:"    << setw(10) << l_twin   << endl
           << "#" << setw(15) << right << "idx_twin:"  << setw(10) << idx_twin << endl
           << "#" << setw(15) << right << "N_step:"    << setw(10) << N_step   << endl;
    output << "#--------------------------------------#" << endl;
}