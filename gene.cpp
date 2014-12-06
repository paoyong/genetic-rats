#include <iostream>
#include <string>
#include <vector>
#include <bitset>
#include <math.h>
#include <algorithm>
#include <stdlib.h>
#include <ctime>
#include <sqlite3.h>
#include <stdio.h>

/* Number of random of rats to initialize */
#define INITIAL_POPULATION          10000

/* Number of generations to run */
#define NUM_GENERATIONS             100

/* How long the chromosome is */
#define CHROMOSOME_LENGTH           290

/* Keith's Reproduce Constants
 * 
 * Good numbers:
 * Lower percent of gene to mutate
 * Very high chance to mutate */
#define MUTATION_RATE_PERCENT       100
#define MUTATION_RATE_OUTOF         100

#define PERCENT_OF_GENE_TO_MUTATE   1
#define GENE_MUTATE_OUTOF           100

#define CROSSOVER_RATE_PERCENT      35
#define NUM_COUPLES_TO_PICK         25
#define NUM_CHILDREN                4

using namespace std;

/* Our domain of ascii characters */
static const char alphanum[] = "*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz";

class Map
{
    public:
        Map();
        Map(string seed);
        int width;
        int height;
        string values;
        char value_at(int x, int y);
        void set_value_at(int x, int y, char val);
        bitset<24> observe(int x, int y);
};

Map::Map(string seed)
{
    string delim = ":";
    int start = 0U;
    int end = seed.find(delim);
    string rows = seed.substr(start, end - start);
    
    height = stoi(rows);

    start  = end + delim.length();
    end = seed.find(delim, start);
    string cols = seed.substr(start, end-start);
    
    width = stoi(cols);

    start  = end + delim.length();
    end = seed.find(delim, start);

    values = seed.substr(start, end);
}

void Map::set_value_at(int x, int y, char val)
{
    if (x < 0 || y < 0 || x >= height || y >= width){
        return;
    }
    values[x*width + y] = val;
    return;
}

char Map::value_at(int x, int y)
{
    if (x < 0 || y < 0 || x >= height || y >= width){
        char temp = '*';
        return temp;
    }
    return values.at(x*width + y);
}

bitset<24> Map::observe(int x, int y)
{
    /*
    *
    * There are 8 zones around the rat, four close zones and four far zones
    * The close zones are the squares in a given quadrant that are 1 away 
    * (e.g. to my left, to my top-left, and above me are the one away squares in
    * the -x, -y  or NW quadrant)
    * The far zones are the squares from each quadrant that are 2 or 3 moves away
    * (e.g. the squares with distance (-3,0), (0, -3), (-3, -3) are the outer 
    * corners of the NW quadrant.
    * The quadrants are always ordered NW then NE then SW then SE first 
    * close and then far.
    * This method outputs 24 0/1 values, first the 8 zones are asked if they have
    * an obstacle "*" those 8 answers will be senses[0] through senses[7] in the 
    * above mentioned order
    * Then each of the 8 zones is asked is they have food, "$", again 8 answers.
    * Finally each zone is asked if it contains a pit "X".
    * These will act as the rat's "senses".
    */
    string targets = "*$X";
    string close[4] = {"","","",""};
    string far[4] = {"","","",""};
    int quadrant[4][2] = {{-1,-1},{-1,1},{1,-1},{1,1}};
    for (int dx = 0; dx < 4; dx++)
    {
        for (int dy = 0; dy < 4; dy++)
        {
            for (int qi = 0; qi < 4; qi++)
            {
                int sign_x = quadrant[qi][0];
                int sign_y = quadrant[qi][1];
                if (dx == 0 && dy == 0){
                    break;
                } else if (dx > 1 || dy > 1){
                    far[qi] += value_at(x + sign_x*dx, y + sign_y*dy);
                } else {
                    close[qi] += value_at(x + sign_x*dx, y + sign_y*dy);
                }
            }
        }
    }
    int counter = 0;
    bitset<24> senses;
    bool bit_value = 0;
    size_t found;
    for (int i = 0; i < 3; i++)
    {
        char target = targets.at(i);
        for (int qi = 0; qi < 4; qi++)
        {
            bit_value = 0;
            found = close[qi].find(target);
            if (string::npos != found)
            {
                bit_value = 1;
            }
            senses.set(counter, bit_value);
            counter++;
        }
        for (int qi = 0; qi < 4; qi++)
        {
            bit_value = 0;
            found = far[qi].find(target);
            if (string::npos != found)
            {
                bit_value = 1;
            }
            senses.set(counter, bit_value);
            counter++;
        }
    }
    return senses;
}

class NeuralNet
{
    public:
        NeuralNet();
        void setWeights(vector<double> weights);
        static vector<double> translateGenome(string genome);
        static string translateWeights(vector<double> weights);
        static const int n_i = 25;
        static const int n_h = 10;
        static const int n_o = 4;
        bitset<n_o> makeChoices(bitset<n_i> inputs);
        double W_ih[n_i][n_h];
        double W_ho[n_h][n_o];
};

NeuralNet::NeuralNet(){};

void NeuralNet::setWeights(vector<double> weights)
{
    /* 
    *  Takes in n_i*n_h + n_h*n_o double values where n_i is the number of input
    * neurons (25 in our case the first is "pain" (did the rat just hit an obstacle?)
    * the other 24 are explained above in Map::observe)
    * The n_h (10 by my settings) hidden neurons are there for complex behavior, 
    * they can kind of be thought of as emotional states.
    * The n_o outputs will be an encoding of the rat's next movement choice.
    */
    int pos = 0;
    int tpos = 0;
    int W_ih_size = n_i*n_h;
    int x, y;
    for (vector<double>::iterator it = weights.begin(); it != weights.end(); ++it){
        if (pos < W_ih_size){
            x = int(pos/n_h);
            y = int(pos % n_h);
            W_ih[x][y] = *it;
            pos++;
        } else {
            tpos = pos - W_ih_size;
            x = int(tpos / n_o);
            y = int(tpos % n_o);
            W_ho[x][y] = *it;
            pos++;
        }
    }
}

bitset<4> NeuralNet::makeChoices(bitset<25> inputs)
{
    bitset<n_h> hidden;
    for (int i = 0; i < n_h; i++)
    {
        double res = 0.0;
        for (int j = 0; j < n_i; j++)
        {
            double tval = double(inputs[j])*double(W_ih[j][i]);
            res += tval;
        }
        /* After checking which weights from inputs affect a given hidden neuron
        * If there is more positive than negative we fire it.
        */
        hidden[i] = res > 0;
    }
    bitset<n_o> output;
    for (int i = 0; i < n_o; i++)
    {
        double res = 0.0;
        for (int j = 0; j < n_h; j++)
        {
            double tval = double(hidden[j])*double(W_ho[j][i]);
            res += tval;
        }
        output[i] = res > 0;
    }
    return output;
}

vector<double> NeuralNet::translateGenome(string genome)
{
    vector<double> temp(0);
    for (int i = 0; i < genome.length(); i++)
    {
        char c = genome.at(i);
        double temp_val = double(c) - 82.0;
        temp.push_back(temp_val/40.0);
    }
    return temp;
}

string NeuralNet::translateWeights(vector<double> weights){
    string answer = "";
    int pos = 0;
    double temp = 0.0;
    for (vector<double>::iterator it = weights.begin(); it != weights.end(); ++it){
        temp = *it;
        temp *= 40;
        temp += 82;
        temp = round(temp);
        answer += char(temp);
    }
    return answer;
}

class Rat
{
    public:
        Rat(int start_x, int start_y, string genome);
        int isDead();
        void changeEnergy(int delta);
        bitset<24> observeWorld(Map world);
        bitset<4> makeChoices(bitset<24> senses);
        void enactChoices(bitset<4> choices);
        int x, y;
        bool hit_obstacle;
        int speed_y, speed_x, energy;
    private:
        string genome;
        NeuralNet brain;
};

Rat::Rat(int start_x, int start_y, string genome)
{
    x = start_x;
    y = start_y;
    speed_x = 1;
    speed_y = 1;
    energy = 30;
    hit_obstacle = 0;
    genome = genome;
    brain.setWeights(NeuralNet::translateGenome(genome));
}

int Rat::isDead()
{
    if (energy <= 0){
        return 1;
    } else {
        return 0;
    }
}

void Rat::changeEnergy(int delta)
{
    energy = energy + delta;
}

void Rat::enactChoices(bitset<4> choices)
{
    speed_x = choices[0] - choices[1];
    speed_y = choices[2] - choices[3];
}

bitset<24> Rat::observeWorld(Map world)
{
    bitset<24> observations = world.observe(x, y);
    return observations;
}

bitset<4> Rat::makeChoices(bitset<24> observations)
{
    bitset<25> pain_and_observations;
    for (int i = 1; i < 25; i++)
    {
        pain_and_observations[i] = observations[i-1];
    }
    pain_and_observations[0] = hit_obstacle;
    bitset<4> choices = brain.makeChoices(pain_and_observations);
    
    /* To see input observations and output decisions uncomment the below code
        cout << "observed "<< pain_and_observations.template to_string<char,
         std::char_traits<char>,
         std::allocator<char> >() << endl;
        cout << "chose to do: "<< choices.template to_string<char,
         std::char_traits<char>,
         std::allocator<char> >() << endl;*/
        /* choices[0] is the choice to move down by 1, choices[1] is to move up by 1, 
        *  choices[2] is to move right by 1, choices[3] is to move left by 1
        * if all four are 1 then the rat doesn't move for example, while if it is 
        * choices[0] = 1, choices[1] = 0, choices[2] = 0 and choices[3] = 1 then it moves 
        * down and left */
    return choices;
}

int simulator(string mapseed, string genome, int start_x, int start_y)
{
    Map board(mapseed);
    Rat arat(start_x, start_y, genome);
    int cur_x, cur_y;
    int target_x, target_y;
    int dx, dy;
    char next_spot;
    int moves = 0;
    while (!arat.isDead())
    {
        moves++;
        cur_x = arat.x;
        cur_y = arat.y;
        /* cout << "rat at "<< cur_x << " by " << cur_y << " on move " << moves << endl; */
        arat.enactChoices(arat.makeChoices(arat.observeWorld(board)));
        target_x = cur_x + arat.speed_x; 
        target_y = cur_y + arat.speed_y; 
        next_spot = board.value_at(target_x, target_y);
        arat.hit_obstacle = 0;
        if (char(next_spot) == char('$'))
        {
            arat.changeEnergy(10);
            board.set_value_at(target_x, target_y, '.');
        } else if (char(next_spot) == char('X'))
        {
            arat.energy = 0;
        } else if (char(next_spot) == char('*'))
        {
            arat.hit_obstacle = 1;
            arat.changeEnergy(-10);
            target_x = cur_x;
            target_y = cur_y;
        }
        arat.changeEnergy(-1);
        arat.x = target_x;
        arat.y = target_y;
    }
    return moves;   
}

/* NOT ANDY'S CODE ANYMORE AFTER THIS LINE!!! */

/* -----------------------------------------------------------------------
 * getRandCharFromGeneDomain
 * 
 * Gets a random character from our alphanum ascii domain of possible gene
 * characters.
 * -----------------------------------------------------------------------
 */
char getRandCharFromGeneDomain()
{
    return alphanum[rand() % (sizeof(alphanum) - 1)];
}

/* -----------------------------------------------------------------------
 * Gene
 * string genome
 * int fitness 
 * 
 * A gene with a fitness level and a genome.
 * -----------------------------------------------------------------------
 */
class Gene
{
    public:
        string genome;
        int fitness;
        void setValues(string g, int f);
        void setFitness(int fit);
};

void Gene::setValues(string g, int f)
{
    genome = g;
    fitness = f;
}

void Gene::setFitness(int fit)
{
    fitness = fit;
}

/* -----------------------------------------------------------------------
 * makeRandomGene():
 *
 * Generate a random CHROMOSOME_LENGTH char ASCII string, that will represent our gene.
 * FORMAT: The format of your genome will be a CHROMOSOME_LENGTH length ascii string 
 * with ascii values between '*' and 'z' (42 to 122).
 * Credits: http://stackoverflow.com/a/440240/4187277
 * -----------------------------------------------------------------------
 */
string makeRandomGene()
{
    char randGene[CHROMOSOME_LENGTH + 1];
    int randNum;

    for (int i = 0; i < CHROMOSOME_LENGTH; i++)
    {
        randGene[i] = getRandCharFromGeneDomain();
    }

    /* Null terminate the string */
    randGene[CHROMOSOME_LENGTH] = 0;
    return randGene;
}

/* -----------------------------------------------------------------------
 * makeGeneVector():
 *
 * Returns a vector of n randomly generated genes of length CHROMOSOME_LENGTH. Gives each
 * object in the vector a fitness level of 0 to start it out.
 * -----------------------------------------------------------------------
 */
vector<Gene> makeGeneVector(int n)
{
    vector<Gene> geneVector;
    
    /* Make INITIAL_POPULATION amount of random genes and push it to firstNGenes */
    for (int i = 0; i < INITIAL_POPULATION; i++)
    {
        Gene g;
        g.setValues(makeRandomGene(), 0);
        geneVector.push_back(g);
    }
    
    return geneVector;
}

/* -----------------------------------------------------------------------
 * runPopThroughMaze():
 *
 * Given sim parameters and a vector<Gene> population of randomly generated
 * rats, run them through the maze and fill up each Gene object in the vector 
 * with a fitness level. Fitness Level is the same as Num Moves survived
 * -----------------------------------------------------------------------
 */
void runPopThroughMaze(string mapS, int startRow, int startCol, vector<Gene>& pop)
{
    for (int i = 0; i < pop.size(); i++)
    {
        int moves = simulator(mapS, pop[i].genome, startRow, startCol);
        if (moves < 40){
            pop[i].setFitness(0);
        }else{
            pop[i].setFitness(moves);
        }
    }
}

/* -----------------------------------------------------------------------
 * rouletteSelect
 * 
 * Given a vector of int weights - select one index for the vector with 
 * a probability based on the weight. The higher the weight, the higher 
 * the chance that index will get selected. NOTE: Must srand(time(NULL))
 * in main ONCE AND ONLY ONCE in the whole program.
 * -----------------------------------------------------------------------
 */
int rouletteSelect(vector<int> weights)
{
    int rouletteSize = 0;
    int roulette[weights.size()];

    for (int i = 0; i < weights.size(); i++)
    {
        rouletteSize += weights[i];
        roulette[i] = rouletteSize;
    }
    
    /* We roll the ball to see where we land on the roulette */
    int throwBall = rand() % rouletteSize;  /* Random number from 0 to rouletteSize */
    int i = 0;
    while (roulette[i] <= throwBall)
    {
        i++;        /* Ball moves to the next roulette slice */
    }

    return i;
}

static int callback(void *data, int argc, char **argv, char **azColName){
    int i;
    fprintf(stderr, "%s: ", (const char*)data);
    for(i=0; i<argc; i++){
        printf("%s = %s\n", azColName[i], argv[i] ? argv[i] : "NULL");
    }
    printf("\n");
    return 0;
}

/* -----------------------------------------------------------------------
 * insertPopulationsIntoDatabase
 *
 * Given a vector<Gene> and a vector<Gene>, inserts the Gene into the population.
 * -----------------------------------------------------------------------
 */
void injectRatIntoPopulation(vector<Gene> & newpop, vector<Gene> & oldpop)
{
    vector<Gene>::iterator it;
    for(it = newpop.begin(); it < newpop.end(); it++)
    {
        oldpop.push_back(*it);
    }
}

/* -----------------------------------------------------------------------
 * insertGenomeIntoDatabase
 *
 * Given a Gene g, inserts it into our local rats.db database using sqlite3.
 * Make sure to compile with "g++ gene.cpp -l sqlite3"!
 * -----------------------------------------------------------------------
 */
void insertGenomeIntoDatabase(Gene g)
{
    /* Database code. Compile with "g++ gene.cpp -l sqlite3" */
    sqlite3 *ratsdb;
    int rc;
    rc = sqlite3_open("rats.db", &ratsdb);
    char *zErrMsg = 0;

    if (rc) {
        fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(ratsdb));
        exit(0);
    } else {
        fprintf(stderr, "Opened rats database successfully\n");
    }

    /* SQL statement */
    string sqlInsertCommand = "INSERT INTO rats (genome, fitness) \
        VALUES ('" + g.genome + "', " + to_string(g.fitness) + ");";
    const char * insertComChar = sqlInsertCommand.c_str();

    /* Execute SQL statement */
    rc = sqlite3_exec(ratsdb, insertComChar, callback, 0, &zErrMsg);
    if(rc != SQLITE_OK){
        fprintf(stderr, "SQL error: %s\n", zErrMsg);
        sqlite3_free(zErrMsg);
    } else {
        cout << "Best rat successfully added to database" << endl;
    }
    sqlite3_close(ratsdb);
}

vector<Gene> grabAllGenesFromDatabase()
{
    vector<Gene> databaseRats;
    /* Database code. Compile with "g++ gene.cpp -l sqlite3" */
    sqlite3 *ratsdb;
    sqlite3_stmt *res;
    int rc = sqlite3_open("rats.db", &ratsdb);
    const char * tail;

    if (rc) {
        fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(ratsdb));
        exit(0);
    } else {
        fprintf(stderr, "Opened rats database successfully\n");
    }

    if (sqlite3_prepare_v2(ratsdb, "SELECT Fitness,Genome FROM rats ORDER BY Fitness", 128, &res, &tail) != SQLITE_OK)
    {
        sqlite3_close(ratsdb);
        fprintf(stderr, "Can't retrieve data: %s\n", sqlite3_errmsg(ratsdb));
    }

    cout << "Reading data..." << endl;

    while(sqlite3_step(res) == SQLITE_ROW)
    {
        Gene g;
        g.fitness = sqlite3_column_text(res, 0)
        g.genome = sqlite3_column_text(res, 1);
    }
    vector<Gene>::iterator it;
    for (it = databaseRats.begin(); it < databaseRats.end(); it++)
    {
        cout << "rat found with fitness of " << *it.fitness << endl;
    }

    sqlite3_finalize(res);
    sqlite3_close(ratsdb);
    return databaseRats;
}
/* -----------------------------------------------------------------------
 * chooseMates():
 *
 * Given a vector<Gene> which represents the population/gene pool, select
 * TWO mates with the Roulette Wheel selection algorithm. So basically each
 * gene is weighted based on its fitness level, and the higher the weight
 * the higher the chances it will be chosen to be mated.
 * 
 * We return those two genes in a struct format.
 * -----------------------------------------------------------------------
 */
struct TwoGenes{
    Gene first;
    Gene second;
};

TwoGenes chooseMates(vector<Gene> & population)
{
    TwoGenes chosenMates;

    /* A vector of the fitnesses of our population vector */
    vector<int> populationFitness;

    int chosenMale = -1;
    int chosenFemale = -1;

    /* For each rat in given population, push their fitness
     * into our populationFitness vector so we can run roulette */
    for (int i = 0; i < population.size(); i++)
    {
        populationFitness.push_back(population[i].fitness);
    }
    
    /* First we select the male mate */
    chosenMale = rouletteSelect(populationFitness);
    chosenMates.first = population[chosenMale];
    population.erase(population.begin() + chosenMale);
    populationFitness.erase(populationFitness.begin() + chosenMale);

    /* Then we select the female mate */
    chosenFemale = rouletteSelect(populationFitness);
    chosenMates.second = population[chosenFemale];
    population.erase(population.begin() + chosenFemale);
    populationFitness.erase(populationFitness.begin() + chosenFemale);

    return chosenMates;
}

string mutate(string oldGene){
    string newGene = oldGene;
    int len = oldGene.length();
    int randomSpot;
    int mutateRoll;

    for (int i = 0; i < len; i++){
        mutateRoll = rand() % GENE_MUTATE_OUTOF;

        if (mutateRoll < PERCENT_OF_GENE_TO_MUTATE){
            randomSpot = rand() % CHROMOSOME_LENGTH;
            newGene[i] = getRandCharFromGeneDomain();
        }
    }

    return newGene;
}

void keithsReproduce(Gene male, Gene female, int numChildren, vector<Gene> & populationToAddChildren)
{
    int crossoverDiceRoll;
    int transferPointRoll;
    Gene mostFitParent;
    int totalfitness = male.fitness + female.fitness;

    /* Find the most fit parent */
    if (male.fitness > female.fitness){
        mostFitParent = male;
    } else {
        mostFitParent = female;
    }

    /* Reproduce! */
    for(int j = 0; j < numChildren; j++) {
        Gene child;
        
        crossoverDiceRoll = rand() % 100;   /* RandInt(0, 100) */

        if (crossoverDiceRoll < CROSSOVER_RATE_PERCENT){
            /* Now randomly find the point at which we are going to do crossing over */
            transferPointRoll = rand() % CHROMOSOME_LENGTH;

            /* Now we copypasta a chunk from male and chunk from female */
            string firstChunk = male.genome.substr(0, transferPointRoll);
            string secondChunk = female.genome.substr(transferPointRoll, CHROMOSOME_LENGTH);
            string newGenome = firstChunk + secondChunk;

            /* Roll for mutation */
            if ((rand() % MUTATION_RATE_OUTOF) <= MUTATION_RATE_PERCENT){
                newGenome = mutate(newGenome);
            }
            child.genome = newGenome;
        } else {
            /* Otherwise we just copy and paste mostFitParent -> child */
            child.genome = mostFitParent.genome;
        }

        populationToAddChildren.push_back(child);
    }
}

vector<Gene> createNewGeneration(vector<Gene> oldPopulation, int numChildren, int numMates){
    vector<Gene> newGeneration;

    for (int m = 0; m < numMates; m++){
        TwoGenes chosenCouple = chooseMates(oldPopulation);
        keithsReproduce(chosenCouple.first, chosenCouple.second, numChildren, newGeneration);
    }

    return newGeneration;
}

double findAverageFitness(vector<Gene> & pool)
{
    int sum = 0;
    double avg;
    for (int i = 0; i < pool.size(); i++)
        sum += pool[i].fitness;
    
    avg = sum/pool.size();
    return avg;
    
}

void keithsReproduceForGenerations(string mapS, int startRow, int startCol, int numGenerations)
{
    /* Our first N randomized genes to get the ball rolling, run it and choose mates */
    vector<Gene> firstNGenes = makeGeneVector(INITIAL_POPULATION);
    vector<Gene> children;
    runPopThroughMaze(mapS, startRow, startCol, firstNGenes);
    cout << "Generation " << 1 << " has population of " << firstNGenes.size() << endl;

    TwoGenes initialMates = chooseMates(firstNGenes);
    
    /* Roll the ball */
    TwoGenes mates = initialMates;
    children = firstNGenes;


    for(int k = 2; k <= numGenerations; k++) {
        children = createNewGeneration(children, NUM_CHILDREN, NUM_COUPLES_TO_PICK);
        runPopThroughMaze(mapS, startRow, startCol, children);
        cout << "Generation " << k << " has population of " << children.size() << " with avg fitness of " << findAverageFitness(children) << endl;
    }
    
    Gene bestGenome;
    bestGenome.fitness = 0;
    for (int i = 0; i < children.size(); i++)
    {
        if (children[i].fitness > bestGenome.fitness) {
            bestGenome.fitness = children[i].fitness;
            bestGenome.genome = children[i].genome;
        }  
    }

    insertGenomeIntoDatabase(bestGenome);
    cout << "-------------------------------------------" << endl;
    cout << "Keith's version" << endl;
    cout << "Best rat has fitness of " << bestGenome.fitness << " after " << numGenerations << " generations" << endl;
    cout << "Its genome is: " << bestGenome.genome << endl;
    cout << "-------------------------------------------" << endl;
}


int main(void){
    /* Generate rand seed. ONLY CALL ONCE IN THE PROGRAM! */
    srand(time(NULL));

    /* IMPORTANT: Make sure to link g++ with sqlite3 with this compile command:
     * g++ gene.cpp -l sqlite3 */

    string mapseed = "25:25:..$.$.X.............X....$X.X*..X$..X...*X$..$...X$.$......X.$.X...XX.$.X*.*.*..X..X.**.......X..$$$...........XX.....................$...X...*.$..X..$X..........$.*..X.....$.X..$*.$X......$...X.*X$......$.**.X.X..XX$X..*....*..X.X....$...X...X........$.X....$...*...X$*........X..$*$$......$$...$*..X.$.$......$.$.$...$..X.*.....X..$......$.XX*..X.$.X......X$*.**.....X*...$..XX..X.....$....X....X...X....X.$X$..X..........$...*.X$..X...$*...........*....XXX$$.$.$..*$XX..XX..*.....$......X.XX$..$$..X$.XX.$$..X.*..*......X......$..$.$$..*...X.........$X....$X.$$.*.$.$.$..**.....X.$.$X.*.$.........$**..X.X.X$X.$.*X.X*..$*.";
    int start_row = 12;
    int start_col = 12;
       
    keithsReproduceForGenerations(mapseed, start_row, start_col, NUM_GENERATIONS);
    vector<Gene> text = grabAllGenesFromDatabase();
    return 0;
}

/*best rat so far: 90

qUZh+Th.+7YX>QL<K,Oz:<Yeqr?1M4khKAGeM[73q0WXs=eAKpIQ,ON,AyfxQAjraO_v/^KzEAVkB9Yq\Qsxq@GAto?c@Tw=50GX+;Jvy4^+72_\ltM=d8,9/Cc<+AyvSfX@Ll0w[U]K[yYN1yVXJd2@P.bW7Uo7DhfAG;r6Yo99PJ+VwmFA+4KD]hnn:7IR=E]saGKYu`K53C`]E=SObw6U.`3p=SJl<HetcK?m;mPS:is1;mQ/POinJQ:p*ry0e?/.VL+qdB-Q[ZO?wnAzvRE]0<+shK?2M>
*/
