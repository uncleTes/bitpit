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

#ifndef __BITPIT_OCTREEPATCH_HPP__
#define __BITPIT_OCTREEPATCH_HPP__

#include <assert.h>
#include <deque>
#include <cstddef>
#include <vector>
#include <unordered_set>

#include "bitpit_PABLO.hpp"
#include "bitpit_patch.hpp"

namespace bitpit {

struct OctreeLevelInfo{
    int    level;
    double h;
    double area;
    double volume;
};

class OctreePatch : public Patch {

public:
	struct OctantInfo {
		OctantInfo() : id(0), internal(true) {};
		OctantInfo(uint32_t _id, bool _internal) : id(_id), internal(_internal) {};

		uint32_t id;
		bool internal;
	};

	OctreePatch(const int &id, const int &dimension, std::array<double, 3> origin,
			double length, double dh);

	~OctreePatch();

	double evalCellVolume(const long &id);
	double evalCellSize(const long &id);
	std::array<double, 3> eval_cell_centroid(const long &id);

	double evalInterfaceArea(const long &id);
	std::array<double, 3> evalInterfaceNormal(const long &id);

	OctantInfo getCellOctant(const long &id) const;
	int getCellLevel(const long &id);

	long getOctantId(const OctantInfo &octantInfo) const;
	const std::vector<uint32_t> & getOctantConnect(const OctantInfo &octantInfo);

	ParaTree & get_tree();

	bool isPointInside(const std::array<double, 3> &point);
	long locatePoint(const std::array<double, 3> &point);

protected:
	const std::vector<Adaption::Info> _update(bool trackAdaption);
	bool _markCellForRefinement(const long &id);
	bool _markCellForCoarsening(const long &id);
	bool _enableCellBalancing(const long &id, bool enabled);

private:
	typedef std::bitset<72> OctantHash;

	struct FaceInfo {
		FaceInfo() : id(Element::NULL_ELEMENT_ID), face(-1) {};
		FaceInfo(long _id, int _face) : id(_id), face(_face) {};

		bool operator==(const FaceInfo &other) const
		{
			return (id == other.id && face == other.face);
		}

		long id;
		int face;
	};

	struct FaceInfoHasher
	{
		std::size_t operator()(const FaceInfo& k) const
		{
			using std::hash;
			using std::string;

			return ((hash<long>()(k.id) ^ (hash<int>()(k.face) << 1)) >> 1);
		}
	};

	typedef std::unordered_set<FaceInfo, FaceInfoHasher> FaceInfoSet;

	std::unordered_map<long, uint32_t, Element::IdHasher> m_cellToOctant;
	std::unordered_map<long, uint32_t, Element::IdHasher> m_cellToGhost;
	std::unordered_map<uint32_t, long> m_octantToCell;
	std::unordered_map<uint32_t, long> m_ghostToCell;

	PabloUniform m_tree;

	std::vector<double> m_tree_dh;
	std::vector<double> m_tree_area;
	std::vector<double> m_tree_volume;

	std::vector<std::array<double, 3> > m_normals;

	bool set_marker(const long &id, const int8_t &value);

	OctantHash evaluate_octant_hash(const OctantInfo &octantInfo);

	std::vector<unsigned long> importOctants(std::vector<OctantInfo> &octantTreeIds);
	std::vector<unsigned long> importOctants(std::vector<OctantInfo> &octantTreeIds, FaceInfoSet &danglingInfoSet);

	FaceInfoSet removeCells(std::vector<long> &cellIds);

	long createVertex(uint32_t treeId);

	long createInterface(uint32_t treeId,
                            std::unique_ptr<long[]> &vertices,
                            std::array<FaceInfo, 2> &faces);

	long createCell(OctantInfo octantInfo,
	                 std::unique_ptr<long[]> &vertices,
	                 std::vector<std::vector<long>> &interfaces,
	                 std::vector<std::vector<bool>> &ownerFlags);
	void deleteCell(long id);
};

}

#endif
