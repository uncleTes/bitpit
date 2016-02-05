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

#ifndef __BITPIT_CARTESIANPATCH_HPP__
#define __BITPIT_CARTESIANPATCH_HPP__

#include <cstddef>
#include <memory>
#include <vector>
#include <array>

#include "bitpit_patch.hpp"

namespace bitpit {

class CartesianPatch : public Patch {

public:
	CartesianPatch(const int &id, const int &dimension, std::array<double, 3> minBB, std::array<double, 3> maxBB, std::array<int,3> nc ); 

	~CartesianPatch();

	double evalCellVolume(const long &id);
	double evalCellSize(const long &id);

	double evalInterfaceArea(const long &id);
	std::array<double, 3> evalInterfaceNormal(const long &id);

protected:
	const std::vector<Adaption::Info> _update(bool trackAdaption);
	bool _markCellForRefinement(const long &id);
	bool _markCellForCoarsening(const long &id);
	bool _enableCellBalancing(const long &id, bool enabled);

private:
	static const int SPACE_MAX_DIM;

	std::array<double,3> m_cellSize;
	std::array<double,3> m_minCoord;
	std::array<double,3> m_maxCoord;

	std::array<int,3> m_nVertices1D;
	std::array<std::vector<double>,3> m_vertexCoord ;

	std::array<int,3> m_nCells1D;
	std::array<std::vector<double>,3> m_cellCoord ;
	double m_cellVolume;

	std::array<int,3> m_interfaceArea;
	std::vector<std::array<double, 3> > m_normals;

	void createVertices();

	void createCells();

	void createInterfaces();
	void createInterfacesDirection(const int &direction);
	int countInterfacesDirection(const int &direction) const;

    std::array<double,3> getSpacing( ) const;
    double getSpacing( const int &d) const;

	long getCellLinearId(const int &i, const int &j, const int &k) const;
	long getCellLinearId(const std::array<int,3> &ijk) const;
	long getCellLinearId(const std::array<double,3> &P)  const;

    std::array<int,3> getCellCartesianId(const long &idx)  const;
    std::array<int,3> getCellCartesianId(const std::array<double,3> &P)  const;

	long getVertexLinearId(const int &i, const int &j, const int &k) const;
	long getVertexLinearId(const std::array<int,3> &ijk) const;
	long getVertexLinearId(const std::array<double,3> &P)  const;

    std::array<int,3> getVertexCartesianId(const long &idx)  const;
    std::array<int,3> getVertexCartesianId(const std::array<double,3> &P)  const;

    long interfaceLinearIndex(const int &normal, const int &i, const int &j, const int &k) const;

    std::vector<int> extractCellSubSet( int const &, int const & ) ;
    std::vector<int> extractCellSubSet( std::array<int,3> const &, std::array<int,3> const & ) ;
    std::vector<int> extractCellSubSet( std::array<double,3> const &, std::array<double,3> const & ) ;

    std::vector<int> extractVertexSubSet( int const &, int const & ) ;
    std::vector<int> extractVertexSubSet( std::array<int,3> const &, std::array<int,3> const & ) ;
    std::vector<int> extractVertexSubSet( std::array<double,3> const &, std::array<double,3> const & ) ;

    bool isPointInGrid( std::array<double,3> const & ) ;
    bool isPointInGrid( std::array<double,3> const &, int &) ;
    bool isPointInGrid( std::array<double,3> const &, std::array<int,3> &) ;
    bool isPointInGrid( std::array<double,3> const &, int &, int &, int &) ;

};

}

#endif
