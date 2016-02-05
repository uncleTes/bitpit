/*---------------------------------------------------------------------------*\
 *
 *  bitpit
 *
 *  Copyright (C) 2015-2016 OPTIMAD engineering Srl
 *
 *  -------------------------------------------------------------------------
 *  License
 *  This file is part of bitbit.
 *
 *  bitpit is free software: you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License v3 (LGPL)
 *  as published by the Free Software Foundation.
 *
 *  bitpit is distributed in the hope that it will be useful, but WITHOUT
 *  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 *  FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
 *  License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with bitpit. If not, see <http://www.gnu.org/licenses/>.
 *
\*---------------------------------------------------------------------------*/

#include <math.h>

#include "bitpit_common.hpp"

#include "cartesianpatch.hpp"

namespace bitpit {

/*!
	\ingroup cartesianpatch
	@{
*/

/*!
	\class CartesianPatch

	\brief The CartesianPatch defines a Cartesian patch.

	CartesianPatch defines a Cartesian patch.
*/

const int CartesianPatch::SPACE_MAX_DIM = 3;

/*!
	Creates a new patch.
*/
CartesianPatch::CartesianPatch(const int &id, const int &dimension, std::array<double, 3> minBB, std::array<double,3> maxBB, std::array<int,3> nc)
	: Patch(id, dimension)
{

	std::cout << ">> Initializing cartesian mesh\n";

	// default initialization
    m_cellSize = {{0,0,0}} ;
    m_nCells1D = {{0,0,0}} ;
    m_nVertices1D = {{0,0,0}} ;
    m_minCoord = {{0,0,0}} ;
    m_maxCoord = {{0,0,0}} ;


    // Global info
    m_cellVolume = 1;
    std::array<double,3> h ;
	for (int n = 0; n < dimension; n++) {
        h[n] = (maxBB[n] -minBB[n]) /(float) nc[n] ;
        m_cellVolume *= h[n] ;
    }


	for (int n = 0; n < dimension; n++) {

		// Mesh Bounding Box
		m_minCoord[n] = minBB[n] ;
		m_maxCoord[n] = maxBB[n] ;

		// Mesh spacing
		m_cellSize[n] = h[n] ;

		// Vertices 
        m_nVertices1D[n] = nc[n]+1 ;

        m_vertexCoord[n].resize( m_nVertices1D[n] ) ;
        for (int i = 0; i < m_nVertices1D[n]; i++) {
            m_vertexCoord[n][i] = m_minCoord[n] + i * m_cellSize[n];
        }

        // Interfaces
        m_interfaceArea[n] = m_cellVolume / m_cellSize[n] ;

        for (int i = -1; i <= 1; i += 2) {
            std::array<double, 3> normal = {{0.0, 0.0, 0.0}};
            normal[n] = i;

            m_normals.push_back(normal);
        }

		// Cells 
        m_nCells1D[n] = nc[n] ;

        m_cellCoord[n].resize( m_nCells1D[n] ) ;
        for (int i = 0; i < m_nCells1D[n]; i++) {
            m_cellCoord[n][i] = m_minCoord[n] + ( (float) i +0.5 ) * m_cellSize[n];
        }


	}


}

/*!
  Destroys the patch.
  */
CartesianPatch::~CartesianPatch()
{
}

/*!
  Evaluates the volume of the specified cell.

  \param id is the id of the cell
  \result The volume of the specified cell.
  */
double CartesianPatch::evalCellVolume(const long &id)
{
    BITPIT_UNUSED(id);

    return m_cellVolume;
}

/*!
  Evaluates the characteristic size of the specified cell.

  \param id is the id of the cell
  \result The characteristic size of the specified cell.
  */
double CartesianPatch::evalCellSize(const long &id)
{
    BITPIT_UNUSED(id);

    return pow(m_cellVolume, 1. / getDimension());
}

/*!
  Evaluates the area of the specified interface.

  \param id is the id of the interface
  \result The area of the specified interface.
  */
double CartesianPatch::evalInterfaceArea(const long &id)
{

    int dir( getInterface(id).getOwnerFace() );
    dir = dir /2; 

    return m_interfaceArea[dir] ;

}

/*!
  Evaluates the normal of the specified interface.

  \param id is the id of the interface
  \result The normal of the specified interface.
  */
std::array<double, 3> CartesianPatch::evalInterfaceNormal(const long &id)
{
    const Interface &interface = getInterface(id);
    int ownerFace = interface.getOwnerFace();

    return m_normals[ownerFace];
}

/*!
  Updates the patch.

  \result Returns a vector of Adaption::Info that can be used to track
  the changes done during the update.
  */
const std::vector<Adaption::Info> CartesianPatch::_update(bool trackAdaption)
{
    if (!isDirty()) {
        return std::vector<Adaption::Info>();
    }

    std::cout << ">> Updating cartesian mesh\n";

    // Reset the mesh
    reset();

    // Definition of the mesh
    createVertices();
    createCells();
    createInterfaces();

    // Adaption info
    std::vector<Adaption::Info> adaptionData;
    if (trackAdaption) {
        adaptionData.emplace_back();
        Adaption::Info &adaptionCellInfo = adaptionData.back();
        adaptionCellInfo.type   = Adaption::TYPE_CREATION;
        adaptionCellInfo.entity = Adaption::ENTITY_CELL;
        adaptionCellInfo.current.reserve(m_cells.size());
        for (auto &cell : m_cells) {
            adaptionCellInfo.current.emplace_back();
            unsigned long &cellId = adaptionCellInfo.current.back();
            cellId = cell.get_id();
        }

        adaptionData.emplace_back();
        Adaption::Info &adaptionInterfaceInfo = adaptionData.back();
        adaptionInterfaceInfo.type   = Adaption::TYPE_CREATION;
        adaptionInterfaceInfo.entity = Adaption::ENTITY_INTERFACE;
        adaptionInterfaceInfo.current.reserve(m_interfaces.size());
        for (auto &interface : m_interfaces) {
            adaptionInterfaceInfo.current.emplace_back();
            unsigned long &interfaceId = adaptionInterfaceInfo.current.back();
            interfaceId = interface.get_id();
        }
    } else {
        adaptionData.emplace_back();
    }

    // Done
    return adaptionData;
}

/*!
  Creates the vertices of the patch.
  */
void CartesianPatch::createVertices()
{
    std::cout << "  >> Creating vertices\n";

    // Definition of the vertices
    long nTotalVertices = 1;
    for (int n = 0; n < getDimension(); n++) {
        nTotalVertices *= m_nVertices1D[n];
    }

    std::cout << "    - Vertex count: " << nTotalVertices << "\n";

    m_vertices.reserve(nTotalVertices);

    for (int k = 0; (isThreeDimensional()) ? (k < m_nVertices1D[Vertex::COORD_Z]) : (k <= 0); k++) {
        for (int j = 0; j < m_nVertices1D[Vertex::COORD_Y]; j++) {
            for (int i = 0; i < m_nVertices1D[Vertex::COORD_X]; i++) {

                long id_vertex = getVertexLinearId(i, j, k);
                Patch::createVertex(id_vertex);
                Vertex &vertex = m_vertices[id_vertex];

                // Coordinate
                std::array<double, 3> coords;
                coords[Vertex::COORD_X] = m_vertexCoord[Vertex::COORD_X][i];
                coords[Vertex::COORD_Y] = m_vertexCoord[Vertex::COORD_Y][j];
                if (isThreeDimensional()) {
                    coords[Vertex::COORD_Z] = m_vertexCoord[Vertex::COORD_Z][k];
                } else {
                    coords[Vertex::COORD_Z] = 0.0;
                }

                vertex.setCoords(coords);
            }
        }
    }
}

/*!
  Creates the cells of the patch.
  */
void CartesianPatch::createCells()
{
    std::cout << "  >> Creating cells\n";

    // Info on the cells
    ElementInfo::Type cellType;
    if (isThreeDimensional()) { //TODO chiedere andrea
        cellType = ElementInfo::VOXEL;
    } else {
        cellType = ElementInfo::PIXEL;
    }

    const ElementInfo &cellTypeInfo = ElementInfo::getElementInfo(cellType);
    const int &nCellVertices = cellTypeInfo.nVertices;

    // Count the cells
    long nTotalCells = 1;
    for (int n = 0; n < getDimension(); n++) {
        nTotalCells *= m_nCells1D[n];
    }

    std::cout << "    - Cell count: " << nTotalCells << "\n";

    m_cells.reserve(nTotalCells);

    // Create the cells
    std::array<double, 3> centroid = {0.0, 0.0, 0.0};
    for (int k = 0; (isThreeDimensional()) ? (k < m_nCells1D[Vertex::COORD_Z]) : (k <= 0); k++) {
        for (int j = 0; j < m_nCells1D[Vertex::COORD_Y]; j++) {
            for (int i = 0; i < m_nCells1D[Vertex::COORD_X]; i++) {
                long id_cell = getCellLinearId(i, j, k);
                Patch::createCell(id_cell);
                Cell &cell = m_cells[id_cell];

                // Initialize the cell
                cell.initialize(cellType, 1);

                // Interior flag
                cell.setInterior(true);

                // Connectivity
                cell.setVertex(0, getVertexLinearId(i,     j,     k));
                cell.setVertex(1, getVertexLinearId(i + 1, j,     k));
                cell.setVertex(2, getVertexLinearId(i,     j + 1, k));
                cell.setVertex(3, getVertexLinearId(i + 1, j + 1, k));
                if (isThreeDimensional()) {
                    cell.setVertex(4, getVertexLinearId(i,     j,     k + 1));
                    cell.setVertex(5, getVertexLinearId(i + 1, j,     k + 1));
                    cell.setVertex(6, getVertexLinearId(i,     j + 1, k + 1));
                    cell.setVertex(7, getVertexLinearId(i + 1, j + 1, k + 1));
                }
            }
        }
    }
}

/*!
  Creates the interfaces of the patch.
  */
void CartesianPatch::createInterfaces()
{
    std::cout << "  >> Creating interfaces\n";

    // Count the interfaces
    long nTotalInterfaces = 0;
    for (int n = 0; n < getDimension(); n++) {
        nTotalInterfaces += countInterfacesDirection(n);
    }

    std::cout << "    - Interface count: " << nTotalInterfaces << "\n";

    // Create the interfaces
    m_interfaces.reserve(nTotalInterfaces);

    createInterfacesDirection(Vertex::COORD_X);
    createInterfacesDirection(Vertex::COORD_Y);
    if (isThreeDimensional()) {
        createInterfacesDirection(Vertex::COORD_Z);
    }
}

/*!
  Counts the interfaces normal to the given direction.

  \param direction the method will count the interfaces normal to this
  direction
  */
int CartesianPatch::countInterfacesDirection(const int &direction) const
{

    int nInterfaces = 1;
    std::array<int,3> interfaceCount1D  = m_nCells1D;

    interfaceCount1D[direction]++ ;

    for (int n = 0; n < getDimension(); n++) {
        nInterfaces *= interfaceCount1D[n];
    }

    return nInterfaces;
}

/*!
  Creates the interfaces normal to the given direction.

  \param direction the method will creat the interfaces normal to this
  direction
  */
void CartesianPatch::createInterfacesDirection(const int &direction)
{
    std::cout << "  >> Creating interfaces normal to direction " << direction << "\n";

    std::array<int,3>   interfaceCount1D;

    interfaceCount1D = m_nCells1D ;
    interfaceCount1D[direction]++ ;

    // Info on the interfaces
    ElementInfo::Type interfaceType;
    if (isThreeDimensional()) {
        interfaceType = ElementInfo::PIXEL;
    } else {
        interfaceType = ElementInfo::LINE;
    }

    const ElementInfo &interfaceTypeInfo = ElementInfo::getElementInfo(interfaceType);
    const int nInterfaceVertices = interfaceTypeInfo.nVertices;

    // Counters
    std::array<int,3> counters = {{0, 0, 0}};
    int &i = counters[Vertex::COORD_X];
    int &j = counters[Vertex::COORD_Y];
    int &k = counters[Vertex::COORD_Z];

    // Creation of the interfaces
    for (k = 0; (isThreeDimensional()) ? (k < interfaceCount1D[Vertex::COORD_Z]) : (k <= 0); k++) {
        for (j = 0; j < interfaceCount1D[Vertex::COORD_Y]; j++) {
            for (i = 0; i < interfaceCount1D[Vertex::COORD_X]; i++) {

                long id_interface = Patch::createInterface();
                Interface &interface = m_interfaces[id_interface];

                // Interface type
                if (isThreeDimensional()) {
                    interface.setType(ElementInfo::PIXEL); //TODO perche PIXEL
                } else {
                    interface.setType(ElementInfo::LINE);
                }

                // Owner
                std::array<int,3> ownerIJK;
                for (int n = 0; n < 3; n++) {
                    ownerIJK[n] = counters[n];
                }
                if (counters[direction] > 0) {
                    ownerIJK[direction] -= 1;
                }
                Cell &owner = m_cells[getCellLinearId(ownerIJK)];

                int ownerFace = 2 * direction;
                if (counters[direction] == 0) {
                    ownerFace++;
                }

                interface.setOwner(owner.get_id(), ownerFace);
                owner.setInterface(ownerFace, 0, interface.get_id());

                // Neighbour
                if (counters[direction] != 0 && counters[direction] != interfaceCount1D[direction] - 1) {
                    std::array<int,3> neighIJK;
                    for (int n = 0; n < SPACE_MAX_DIM; n++) {
                        neighIJK[n] = counters[n];
                    }

                    Cell &neigh = m_cells[getCellLinearId(neighIJK)];

                    int neighFace = 2 * direction + 1;

                    interface.setNeigh(neigh.get_id(), neighFace);
                    neigh.setInterface(neighFace, 0, interface.get_id());
                } else {
                    interface.unsetNeigh();
                }

                // Connectivity
                std::unique_ptr<long[]> connect = std::unique_ptr<long[]>(new long[nInterfaceVertices]);
                if (direction == Vertex::COORD_X) {
                    connect[0] = getVertexLinearId(i, j,     k);
                    connect[1] = getVertexLinearId(i, j + 1, k);
                    if (isThreeDimensional()) {
                        connect[2] = getVertexLinearId(i, j + 1, k + 1);
                        connect[3] = getVertexLinearId(i, j,     k + 1);
                    }
                } else if (direction == Vertex::COORD_Y) {
                    connect[0] = getVertexLinearId(i,     j,     k);
                    connect[1] = getVertexLinearId(i + 1, j,     k);
                    if (isThreeDimensional()) {
                        connect[2] = getVertexLinearId(i + 1, j, k + 1);
                        connect[3] = getVertexLinearId(i,     j, k + 1);
                    }
                } else if (direction == Vertex::COORD_Z) {
                    connect[0] = getVertexLinearId(i,     j,     k);
                    connect[1] = getVertexLinearId(i + 1, j,     k);
                    if (isThreeDimensional()) {
                        connect[2] = getVertexLinearId(i + 1, j + 1, k);
                        connect[3] = getVertexLinearId(i,     j + 1, k);
                    }
                }

                interface.setConnect(std::move(connect));
            }
        }
    }
}

/*!
  Marks a cell for refinement.

  This is a void function since mesh refinement is not implemented
  for Cartesian meshes.

  \param id is the id of the cell that needs to be refined
  */
bool CartesianPatch::_markCellForRefinement(const long &id)
{
    BITPIT_UNUSED(id);

    return false;
}

/*!
  Marks a cell for coarsening.

  This is a void function since mesh coarsening is not implemented
  for Cartesian meshes.

  \param id the cell to be refined
  */
bool CartesianPatch::_markCellForCoarsening(const long &id)
{
    BITPIT_UNUSED(id);

    return false;
}

/*!
  Enables cell balancing.

  This is a void function since mesh coarsening is not implemented
  for Cartesian meshes.

  \param id is the id of the cell
  \param enabled defines if enable the balancing for the specified cell
  */
bool CartesianPatch::_enableCellBalancing(const long &id, bool enabled)
{
    BITPIT_UNUSED(id);
    BITPIT_UNUSED(enabled);

    return false;
}


/*!
  Get cell spacing in all directions;
  */
std::array<double,3> CartesianPatch::getSpacing( ) const
{
    return m_cellSize;
}

/*!
  Get cell spacing in one directions;
  @param[in] d direction
  */
double CartesianPatch::getSpacing( const int &d ) const
{
    return m_cellSize[d];
}
/*!
  Converts the cell cartesian notation to a linear notation
  */
long CartesianPatch::getCellLinearId(const int &i, const int &j, const int &k) const
{
    long id = i;
    id += m_nCells1D[Vertex::COORD_X] * j;
    if (getDimension() == 3) {
        id += m_nCells1D[Vertex::COORD_Y] * m_nCells1D[Vertex::COORD_X] * k;
    }

    return id;
}

/*!
  Converts the cell cartesian notation to a linear notation
  */
long CartesianPatch::getCellLinearId(const std::array<int,3> &ijk) const
{
    return getCellLinearId(ijk[Vertex::COORD_X], ijk[Vertex::COORD_Y], ijk[Vertex::COORD_Z]);
}

/*!
  Returns linear indices of cell containing the point.
  */
long CartesianPatch::getCellLinearId( std::array<double,3> const &P) const
{
    return getCellLinearId( getCellCartesianId(P) );
}

/*! 
 * Compute the cartesian indices from the linear index
 * @param[in] idx linear index
 * @return cartesian cell indices
 */
std::array<int,3> CartesianPatch::getCellCartesianId( long const &idx ) const
{

    std::array<int,3>    id;
    int ijPlane(  m_nCells1D[0]*m_nCells1D[1] ) ;

    id[0] = idx % m_nCells1D[0] ;
    id[2] = idx / ijPlane;
    id[1] =  (idx - id[2] *ijPlane ) /m_nCells1D[0]  ;


    return id; 
};

/*! 
 * Compute the cartesian indices of cell which contains given point.
 * If point is outside mesh closest cell is returned.
 * @param[in]  P       point coordinates
 * @return             cartesian cell indices
 */
std::array<int,3> CartesianPatch::getCellCartesianId( std::array<double,3> const &P ) const
{

    int         d ;
    std::array<int,3>    id;

    id.fill(0) ;

    for( d=0; d<getDimension(); ++d){
        id[d] = std::min( m_nCells1D[d]-1, std::max(0, (int) floor( (P[d] - m_minCoord[d])/m_cellSize[d] )) );
    };

    return id; 
};

/*!
  Converts the vertex cartesian notation to a linear notation
  */
long CartesianPatch::getVertexLinearId(const int &i, const int &j, const int &k) const
{
    long id = i;
    id += m_nVertices1D[Vertex::COORD_X] * j;
    if (getDimension() == 3) {
        id += m_nVertices1D[Vertex::COORD_Y] * m_nVertices1D[Vertex::COORD_X] * k;
    }

    return id;
}

/*!
 * Compute the linear index of closest vertex to given point.
 * @param[in] P point coordinates
 * @return cartesian indices
 */
long CartesianPatch::getVertexLinearId( std::array<double,3> const &P ) const
{
    return getVertexLinearId( getVertexCartesianId(P) );
}

/*!
  Converts the vertex cartesian notation to a linear notation
  */
long CartesianPatch::getVertexLinearId(const std::array<int,3> &ijk) const
{
    return getVertexLinearId(ijk[Vertex::COORD_X], ijk[Vertex::COORD_Y], ijk[Vertex::COORD_Z]);
}

/*!
 * Transformation from linear index to cartesian indices.
 * No check on bounds is performed
 * @param[in] idx node linear index
 * @return node cartesian indices
 */
std::array<int,3> CartesianPatch::getVertexCartesianId( long const &idx ) const
{

    // Local variables
    std::array<int,3>    id ;
    int ijPlane(  m_nVertices1D[0]*m_nVertices1D[1] ) ;

    id[0] = idx % m_nVertices1D[0] ;
    id[2] = idx / ijPlane ;
    id[1] =  (idx - id[2] *ijPlane ) /m_nVertices1D[0]  ;

    return id; 

};

/*! 
 * Compute the cartesian indices of closest node to given point.
 * @param[in] P point coordinates
 * @return cartesian indices
 */
std::array<int,3> CartesianPatch::getVertexCartesianId( std::array<double,3> const &P ) const
{
    int         d ;
    std::array<int,3>    id;

    id.fill(0) ;

    for( d=0; d<getDimension(); ++d){
        id[d] = std::min( m_nVertices1D[d]-1, std::max(0, (int) round( (P[d] - m_minCoord[d])/m_cellSize[d] )) );
    };

    return id; 
};

/*!
  Converts the interface cartesian notation to a linear notation
  */
long CartesianPatch::interfaceLinearIndex(const int &normal, const int &i, const int &j, const int &k) const
{
    std::array<int,3> nInterfaces(m_nCells1D) ;
    nInterfaces[normal]++ ;

    long id(0);

    for( int d=0; d<normal; ++d){
        id += countInterfacesDirection(d) ;
    };

    id += k *nInterfaces[0] *nInterfaces[1] + j *nInterfaces[0] +i ;

    return id;
}

/*! 
 * Calculate cell subset indices form cartesian indices
 * @param[in] i0 min cartesian indices
 * @param[in] i1 max cartesian indices
 * @return cell linear indices of subset mesh
 */
std::vector<int> CartesianPatch::extractCellSubSet( std::array<int,3> const &i0, std::array<int,3> const &i1 ){

    int                     i, j, k; 
    std::vector<int>               ids;
    std::vector<int>::iterator     it;

    i  =  i1[0]-i0[0]+1  ;
    j  =  i1[1]-i0[1]+1  ;
    k  =  i1[2]-i0[2]+1  ;

    i  =  i *j *k ;
    ids.resize(i) ;

    it = ids.begin() ;

    for( k=i0[2]; k<=i1[2]; ++k){
        for( j=i0[1]; j<=i1[1]; ++j){
            for( i=i0[0]; i<=i1[0]; ++i){

                *it = getCellLinearId( i, j, k) ;            
                ++it ;

            };
        };
    };

    return ids; 

};

/*! 
 * Calculate cell subset indices form linear indices
 * @param[in] I0 min linear indices
 * @param[in] I1 max linear indices
 * @return cell linear indices of subset mesh
 */
std::vector<int> CartesianPatch::extractCellSubSet( int const &I0, int const &I1 ){

    return extractCellSubSet( getCellCartesianId(I0), getCellCartesianId(I1) ); 

};

/*! 
 * Calculate cell subset indices form min and max point.
 * The cell conataining the points (or closest to them) are maintained
 * @param[in] P0 min point
 * @param[in] P1 max point
 * @return cell linear indices of subset mesh
 */
std::vector<int> CartesianPatch::extractCellSubSet( std::array<double,3> const &P0, std::array<double,3> const &P1 ){

    return extractCellSubSet( getCellCartesianId(P0), getCellCartesianId(P1) ); 

};

/*! 
 * Calculate vertex subset indices form cartesian indices
 * @param[in] i0 min cartesian indices
 * @param[in] i1 max cartesian indices
 * @return node linear indices of subset mesh
 */
std::vector<int> CartesianPatch::extractVertexSubSet( std::array<int,3> const &i0, std::array<int,3> const &i1 ){

    int                     i, j, k; 
    std::vector<int>               ids;
    std::vector<int>::iterator     it;

    i  =  i1[0]-i0[0]+1  ;
    j  =  i1[1]-i0[1]+1  ;
    k  =  i1[2]-i0[2]+1  ;

    i  =  i *j *k ;
    ids.resize(i) ;

    it = ids.begin() ;

    for( k=i0[2]; k<=i1[2]; ++k){
        for( j=i0[1]; j<=i1[1]; ++j){
            for( i=i0[0]; i<=i1[0]; ++i){

                *it = getVertexLinearId( i, j, k) ;            
                ++it ;

            };
        };
    };

    return ids; 

};

/*! 
 * Calculate vertex subset indices form linear indices
 * @param[in] I0 min linear indices
 * @param[in] I1 max linear indices
 * @return vertex linear indices of subset mesh
 */
std::vector<int> CartesianPatch::extractVertexSubSet( int const &I0, int const &I1 ){

    return extractVertexSubSet( getVertexCartesianId(I0), getVertexCartesianId(I1) ); 

};

/*! 
 * Calculate vertex subset indices form min and max point.
 * The vertex closest to the points are used as limites 
 * @param[in] P0 min point
 * @param[in] P1 max point
 * @return vertex linear indices of subset mesh
 */
std::vector<int> CartesianPatch::extractVertexSubSet( std::array<double,3> const &P0, std::array<double,3> const &P1 ){

    return extractVertexSubSet( getVertexCartesianId(P0), getVertexCartesianId(P1) ); 

};

/*! 
 * Check if point lies within the mesh
 * @param[in]  P point to bechecked
 * @return true if point lies within grid
 */
bool CartesianPatch::isPointInGrid(  std::array<double,3> const &P ){

    int     d;

    for( d=0; d<getDimension(); ++d){

        if( P[d]< m_minCoord[d] || P[d] > m_maxCoord[d] ){
            return false;
        };
    };

    return true ;
};

/*! 
 * Check if point lies within the mesh
 * @param[in] P point to bechecked
 * @param[out] I cartesian indices of cell containing the point      
 * @return true if point lies within grid
 */
bool CartesianPatch::isPointInGrid( std::array<double,3> const &P, std::array<int,3> &I){

    int     d;

    for( d=0; d<getDimension(); ++d){

        if( P[d]< m_minCoord[d] || P[d] > m_maxCoord[d] ){
            return false;
        };
    };

    I = getCellCartesianId(P) ;

    return true ;
};

/*! 
 * Check if point lies within the mesh
 * @param[in] P point to be checked      
 * @param[out] i first cartesian index of cell containing the point      
 * @param[out] j second cartesian index of cell containing the point      
 * @param[out] k third cartesian index of cell containing the point      
 * @return true if point lies within grid
 */
bool CartesianPatch::isPointInGrid( std::array<double,3> const &P, int &i, int &j, int &k ){

    bool        inGrid ;
    std::array<int,3>    I;

    if(isPointInGrid(P,I) ){

        i = I[0] ;
        j = I[1] ;
        k = I[2] ;

        return true ;
    } ;

    return false ;
};

/*! 
 * Check if point lies within the mesh
 * @param[in] P point to be checked
 * @param[out] I linear index of cell containing the point      
 * @return true if point lies within grid
 */
bool CartesianPatch::isPointInGrid( std::array<double,3> const &P, int &I ){

    int     d;

    for( d=0; d<getDimension(); ++d){

        if( P[d]< m_minCoord[d] || P[d] > m_maxCoord[d] ){
            return false;
        };
    };

    I = getCellLinearId(P) ;

    return true ;
};

/*!
  @}
  */

}
