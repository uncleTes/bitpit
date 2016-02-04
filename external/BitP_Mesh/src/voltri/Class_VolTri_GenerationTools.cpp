// ========================================================================== //
//                         - Class_VolTri -                                   //
//                                                                            //
// Grid manager for unstructured volume meshes.                               //
// ========================================================================== //
// INFO                                                                       //
// ========================================================================== //
// Author   : Alessandro Alaia                                                //
// Version  : v2.0                                                            //
//                                                                            //
// All rights reserved.                                                       //
// ========================================================================== //

// ========================================================================== //
// INCLUDES                                                                   //
// ========================================================================== //
# include "Class_VolTri.hpp"

// ========================================================================== //
// IMPLEMENTATIONS                                                            //
// ========================================================================== //

// ADDING TOOLS ============================================================= //

// -------------------------------------------------------------------------- //
void Class_VolTri::AddVertex(
    a3vector1D      &V
) {

// ========================================================================== //
// void Class_VolTri::AddVertex(                                              //
//     a3vector1D      &V)                                                    //
//                                                                            //
// Add a vertex to vertex list.                                               //
// ========================================================================== //
// INPUT                                                                      //
// ========================================================================== //
// - V     : a3vector1D, vertex coordinates. V[0], V[1], ... are              //
//           the x, y, ... coordinates of vertex to be added.                 //
// ========================================================================== //
// OUTPUT                                                                     //
// ========================================================================== //
// - none                                                                     //
// ========================================================================== //

// ========================================================================== //
// VARIABLES DECLARATION                                                      //
// ========================================================================== //

// Local variables
// - none

// Counters
// - none


// ========================================================================== //
// ADD VERTEX                                                                 //
// ========================================================================== //

// Add vertex to the vertex coordinate list
if (Vertex.size() > nVertex) {
    Vertex[nVertex] = V;
}
else {
    Vertex.push_back(V);
}

// Update the number of vertexes
nVertex++;

return; }

// -------------------------------------------------------------------------- //
void Class_VolTri::AddVertex(
    dvector1D       &V
) {

// ========================================================================== //
// void Class_VolTri::AddVertex(                                              //
//     dvector1D       &V)                                                    //
//                                                                            //
// Add a vertex to vertex list.                                               //
// ========================================================================== //
// INPUT                                                                      //
// ========================================================================== //
// - V     : dvector1D, vertex coordinates. V[0], V[1], ... are               //
//           the x, y, ... coordinates of vertex to be added.                 //
// ========================================================================== //
// OUTPUT                                                                     //
// ========================================================================== //
// - none                                                                     //
// ========================================================================== //

// ========================================================================== //
// VARIABLES DECLARATION                                                      //
// ========================================================================== //

// Local variables
a3vector1D              tmp;

// Counters
int                     i, n = min(V.size(), (size_t) 3);


// ========================================================================== //
// ADD VERTEX                                                                 //
// ========================================================================== //

// Store vector into array
for (i = 0; i < n; ++i) {
    tmp[i] = V[i];
} //next i

// Add vertex to the vertex coordinate list
if (Vertex.size() > nVertex) {
    Vertex[nVertex] = tmp;
}
else {
    Vertex.push_back(tmp);
}

// Update the number of vertexes
nVertex++;

return; }

// -------------------------------------------------------------------------- //
void Class_VolTri::AddVertices(
    a3vector2D      &V
) {

// ========================================================================== //
// void Class_VolTri::AddVertices(                                            //
//     a3vector2D      &V)                                                    //
//                                                                            //
// Add multiple vertexes to vertex list                                       //
// ========================================================================== //
// INPUT                                                                      //
// ========================================================================== //
// - V     : a3vector2D, vertex coordinates list. V[i][0], V[i][1], ... are   //
//           the x, y, ... coordinates of the i-th vertex to be added.        //
// ========================================================================== //
// OUTPUT                                                                     //
// ========================================================================== //
// - none                                                                     //
// ========================================================================== //

// ========================================================================== //
// VARIABLES DECLARATION                                                      //
// ========================================================================== //

// Local variables
// none

// Counters
int    i, n = V.size();


// ========================================================================== //
// ADD VERTEX                                                                 //
// ========================================================================== //
for (i = 0; i < n; i++) {
    AddVertex(V[i]);
} //next i

return; }

// -------------------------------------------------------------------------- //
void Class_VolTri::AddVertices(
    dvector2D       &V
) {

// ========================================================================== //
// void Class_VolTri::AddVertices(                                            //
//     dvector2D       &V)                                                    //
//                                                                            //
// Add multiple vertexes to vertex list                                       //
// ========================================================================== //
// INPUT                                                                      //
// ========================================================================== //
// - V     : dvector2D, vertex coordinates list. V[i][0], V[i][1], ... are    //
//           the x, y, ... coordinates of the i-th vertex to be added.        //
// ========================================================================== //
// OUTPUT                                                                     //
// ========================================================================== //
// - none                                                                     //
// ========================================================================== //

// ========================================================================== //
// VARIABLES DECLARATION                                                      //
// ========================================================================== //

// Local variables
// none

// Counters
int    i, n = V.size();


// ========================================================================== //
// ADD VERTEX                                                                 //
// ========================================================================== //
for (i = 0; i < n; i++) {
    AddVertex(V[i]);
} //next i

return; }

// -------------------------------------------------------------------------- //
void Class_VolTri::AddSimplex(
    ivector1D       &E,
    int              s_type
) {

// ========================================================================== //
// void Class_VolTri::AddSimplex(                                             //
//     ivector1D       &E,                                                    //
//     int              s_type)                                               //
//                                                                            //
// Add simplex to simplex list                                                //
// ========================================================================== //
// INPUT                                                                      //
// ========================================================================== //
// - E     : ivector1D, simplex-vertex connectivity                           //
// ========================================================================== //
// OUTPUT                                                                     //
// ========================================================================== //
// - none                                                                     //
// ========================================================================== //

// ========================================================================== //
// VARIABLES DECLARATION                                                      //
// ========================================================================== //

// Local variables
// none

// Counters
// none

// ========================================================================== //
// ADD SIMPLEX                                                                //
// ========================================================================== //

// Add simplex
if (Simplex.size() > nSimplex) {
    Simplex[nSimplex] = E;
    e_type[nSimplex] = s_type;
}
else {
    Simplex.push_back(E);
    e_type.push_back(s_type);
}

// Update number of simplex
nSimplex++;

return; };

// -------------------------------------------------------------------------- //
void Class_VolTri::AddSimplex(
    a3vector2D      &V,
    int              s_type
) {

// ========================================================================== //
// void Class_VolTri::AddSimplex(                                             //
//     a3vector2D      &V,                                                    //
//     int              s_type)                                               //
//                                                                            //
// Add simplex to simplex list                                                //
// ========================================================================== //
// INPUT                                                                      //
// ========================================================================== //
// - V      : a3vector2D, vertex of simplex to be added                       //
// - s_type : int, simplex type id.                                           //
// ========================================================================== //
// OUTPUT                                                                     //
// ========================================================================== //
// - none                                                                     //
// ========================================================================== //

// ========================================================================== //
// VARIABLES DECLARATION                                                      //
// ========================================================================== //

// Local variables
int             n = V.size();
ivector1D       dummy(V.size(),-1);

// Counters
int             i;

// ========================================================================== //
// ADD SIMPLES                                                                //
// ========================================================================== //

// Add simplex vertexes
AddVertices(V);

// Add simplex
n = V.size();
for (i = 0; i < n; i++) {
    dummy[i] = nVertex - n + i;
} //next i
AddSimplex(dummy, s_type);

return; };

// -------------------------------------------------------------------------- //
void Class_VolTri::AddSimplicies(
    ivector2D       &E,
    ivector1D       &s_type
) {

// ========================================================================== //
// void Class_VolTri::AddSimplicies(                                          //
//     ivector2D       &E,                                                    //
//     ivector1D       &s_type)                                               //
//                                                                            //
// Add multiple simplicies to simplex list.                                   //
// ========================================================================== //
// INPUT                                                                      //
// ========================================================================== //
// - E      : ivector2D, simplex-vertex connectivity for each simplex         //
//            to be added                                                     //
// - s_type : ivector1D, simplex types                                        //
// ========================================================================== //
// OUTPUT                                                                     //
// ========================================================================== //
// - none                                                                     //
// ========================================================================== //

// ========================================================================== //
// VARIABLES DECLARATION                                                      //
// ========================================================================== //

// Local variables
// none

// Counters
int           i;

// ========================================================================== //
// ADD SIMPLICIES                                                             //
// ========================================================================== //

// Add simplicies
for (i = 0; i < E.size(); i++) {
    AddSimplex(E[i], s_type[i]);
} //next i

return; }

// -------------------------------------------------------------------------- //
void Class_VolTri::AddSimplicies(
    a3vector2D      &V,
    ivector2D       &E,
    ivector1D       &s_type
) {

// ========================================================================== //
// void Class_VolTri::AddSimplicies(                                          //
//     a3vector2D      &V,                                                    //
//     ivector2D       &E,                                                    //
//     ivector1D       &s_type)                                               //
//                                                                            //
// Add multiple simplicies to simplex list.                                   //
// ========================================================================== //
// INPUT                                                                      //
// ========================================================================== //
// - V      : a3vector2D, vertex coordinate list                              //
// - E      : ivector2D, simplex-vertex connectivity for each simplex         //
//            to be added                                                     //
// - s_type : int, simplex type id.                                           //
// ========================================================================== //
// OUTPUT                                                                     //
// ========================================================================== //
// - none                                                                     //
// ========================================================================== //

// ========================================================================== //
// VARIABLES DECLARATION                                                      //
// ========================================================================== //

// Local variables
int           v_off = nVertex;
int           m, n = E.size();
ivector1D     idummy1D;

// Counters
int           i, j;

// ========================================================================== //
// ADD SIMPLES                                                                //
// ========================================================================== //

// Add vertices ------------------------------------------------------------- //
AddVertices(V);

// Add simplicies ----------------------------------------------------------- //
for (i = 0; i < n; i++) {
    m = E[i].size();
    idummy1D.resize(m, -1);
    for (j = 0; j < m; j++) {
        idummy1D[j] = E[i][j] + v_off;
    } //next j
    AddSimplex(idummy1D, s_type[i]);
} //next i

return; }

// -------------------------------------------------------------------------- //
void Class_VolTri::SetAdjacency(
    int              S,
    ivector1D       &adj
) {

// ========================================================================== //
// void Class_VolTri::SetAdjacency(                                           //
//     int              S,                                                    //
//     ivector1D       &adj)                                                  //
//                                                                            //
// Set adjacency for a given simplex.                                         //
// ========================================================================== //
// INPUT                                                                      //
// ========================================================================== //
// - S         : int, simplex global index                                    //
// - adj       : ivector1D, adjacency list                                    //
// ========================================================================== //
// OUTPUT                                                                     //
// ========================================================================== //
// - none                                                                     //
// ========================================================================== //

// ========================================================================== //
// VARIABLES DECLARATION                                                      //
// ========================================================================== //

// Local variables
// none

// Counters
// none

// ========================================================================== //
// UPDATE ADJACENCY                                                           //
// ========================================================================== //
if (S >= Adjacency.size()) {
    ReshapeAdjacency();
}
if (S < nSimplex) {
    Adjacency[S] = adj;
}

return; };

// -------------------------------------------------------------------------- //
void Class_VolTri::Append(
    Class_VolTri    &Source
) {

// ========================================================================== //
// void Class_VolTri::Append(                                                 //
//     Class_VolTri    &Source)                                               //
//                                                                            //
// Append mesh in Source to the present mesh                                  //
// ========================================================================== //
// INPUT                                                                      //
// ========================================================================== //
// - Source   : Class_VolTri, source mesh to be appended                      //
// ========================================================================== //
// OUTPUT                                                                     //
// ========================================================================== //
// - none                                                                     //
// ========================================================================== //

// ========================================================================== //
// VARIABLES DECLARATION                                                      //
// ========================================================================== //

// Local variables
int        v_off, s_off;

// Counters
int        i, j, m;

// ========================================================================== //
// INTIIALIZE PARAMETERS                                                      //
// ========================================================================== //

// Vertex offset
v_off = nVertex;

// Simplex offset
s_off = nSimplex;

// ========================================================================== //
// APPEND SIMPLEX IN SOURCE TO THE PRESENT TASSELATION                        //
// ========================================================================== //

// Vertex ------------------------------------------------------------------- //
if (Source.nVertex > 0) {

    // Resize vertex list
    Vertex.resize(nVertex + Source.nVertex);

    // Add vertexes
    AddVertices(Source.Vertex);
}

// Simplex ------------------------------------------------------------------ //
if (Source.nSimplex > 0) {

    // Resize simplex list
    Simplex.resize(nSimplex + Source.nSimplex);

    // Add simplex
    for (i = 0; i < Source.nSimplex; i++) {
        Simplex[i + s_off] = Source.Simplex[i] + v_off;
        e_type[i + s_off] = e_type[i];
        nSimplex++;
    } //next i
}

// Adjacencies -------------------------------------------------------------- //
if ((Source.Adjacency.size() > 0) && (Source.Adjacency.size() >= Source.nSimplex)) {

    // Resize adjacency list
    Adjacency.resize(nSimplex);

    // Add adjacencies
    for (i = 0; i < Source.nSimplex; i++) {
        Adjacency[i + s_off].resize(Source.Adjacency[i].size());
        m = Source.Adjacency[i].size();
        for (j = 0; j < m; j++) {
            if (Source.Adjacency[i][j] > 0) {
                Adjacency[i + s_off][j] = Source.Adjacency[i][j] + s_off;
            }
            else {
                Adjacency[i + s_off][j] = -1;
            }
        } //next j
    } //next i
}

return; };

// TRANSFORMATION TOOLS ===================================================== //

// -------------------------------------------------------------------------- //
void Class_VolTri::Scale(
    double           sx,
    double           sy,
    double           sz
) {

// ========================================================================== //
// void Class_VolTri::Scale(                                                  //
//     double           sx,                                                   //
//     double           sy,                                                   //
//     double           sz)                                                   //
//                                                                            //
// Scale mesh along x, y, z axis by factor sx, sy and sz                      //
// respectively.                                                              //
// ========================================================================== //
// INPUT                                                                      //
// ========================================================================== //
// - sx     : double, scaling factor along x direction                        //
// - sy     : double, scaling factor along y direction                        //
// - sz     : double, scaling factor along z direction                        //
// ========================================================================== //
// OUTPUT                                                                     //
// ========================================================================== //
// - none                                                                     //
// ========================================================================== //

// ========================================================================== //
// VARIABLES DECLARATION                                                      //
// ========================================================================== //

// Local variables
// none

// Counters
int      i;

// ========================================================================== //
// SCALE TASSELATION                                                          //
// ========================================================================== //

// Vertex ------------------------------------------------------------------- //
for (i = 0; i < nVertex; i++) {
    Vertex[i][0] = sx*Vertex[i][0];
    Vertex[i][1] = sy*Vertex[i][1];
    Vertex[i][2] = sz*Vertex[i][2];
} //next i

return; };

// -------------------------------------------------------------------------- //
void Class_VolTri::Translate(
    double           sx,
    double           sy,
    double           sz
) {

// ========================================================================== //
// void Class_VolTri::Translate(                                              //
//     double           sx,                                                   //
//     double           sy,                                                   //
//     double           sz)                                                   //
//                                                                            //
// Translate mesh along x, y, z by sx, sy and sz (respectively)               //
// ========================================================================== //
// INPUT                                                                      //
// ========================================================================== //
// - sx      : double, translation along x direction                          //
// - sy      : double, translation along y direction                          //
// - sz      : double, translation along z direction                          //
// ========================================================================== //
// OUTPUT                                                                     //
// ========================================================================== //
// - none                                                                     //
// ========================================================================== //

// ========================================================================== //
// VARIABLES DECLARATION                                                      //
// ========================================================================== //

// Local variables
// none


// Counters
int       i;

// ========================================================================== //
// TRANSLATE VERTEX                                                           //
// ========================================================================== //
for (i = 0; i < nVertex; i++) {
    Vertex[i][0] += sx;
    Vertex[i][1] += sy;
    Vertex[i][2] += sz;
} //next i

return; };

