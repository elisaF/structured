#include <vector>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <boost/property_map/property_map.hpp>
#include <boost/graph/adjacency_list.hpp>

#include "../../edmonds-alg/src/edmonds_optimum_branching.hpp"
#include "MaxSpanTree.h"

// Define a directed graph type that associates a weight with each
// edge. We store the weights using internal properties as described
// in BGL.
typedef boost::property<boost::edge_weight_t, double>       EdgeProperty;
typedef boost::adjacency_list<boost::listS,
boost::vecS,
boost::directedS,
boost::no_property,
EdgeProperty>                 Graph;
typedef boost::graph_traits<Graph>::vertex_descriptor       Vertex;
typedef boost::graph_traits<Graph>::edge_descriptor         Edge;

using namespace trees;

MaxSpanTree::MaxSpanTree()
{
}

MaxSpanTree::~MaxSpanTree()
{
}

std::vector< std::vector<double> > MaxSpanTree::get_tree(std::vector< std::vector<double> > str_scores)
{
    int rows = str_scores.size();
    int cols = str_scores[0].size();

    std::vector< std::vector<double> > tree;

    // Graph with N vertices
    Graph G(cols);

    // Create a vector to keep track of all the vertices and enable us
    // to index them. As a side note, observe that this is not
    // necessary since Vertex is probably an integral type. However,
    // this may not be true of arbitrary graphs and I think this code
    // is a better illustration of a more general case.
    std::vector<Vertex> the_vertices;
    BOOST_FOREACH (Vertex v, vertices(G))
    {
        the_vertices.push_back(v);
    }

    // add edges with weights to the graph
    for (int i=0; i<cols; i++){
        for (int j=1; j<cols; j++) {
            if (i==j){
                continue;
            }
            else {
                add_edge(the_vertices[i], the_vertices[j], str_scores[j-1][i], G);
            }
        }
    }

    // This is how we can get a property map that gives the weights of
    // the edges.
    boost::property_map<Graph, boost::edge_weight_t>::type weights =
            get(boost::edge_weight_t(), G);

    // This is how we can get a property map mapping the vertices to
    // integer indices.
    boost::property_map<Graph, boost::vertex_index_t>::type vertex_indices =
            get(boost::vertex_index_t(), G);

    // Find the maximum branching.
    std::vector<Edge> branching;
    edmonds_optimum_branching<true, false, false>(G,
                                                  vertex_indices,
                                                  weights,
                                                  static_cast<Vertex *>(0),
                                                  static_cast<Vertex *>(0),
                                                  std::back_inserter(branching));

    // Get the edges of the maximum branching
    BOOST_FOREACH (Edge e, branching)
    {
        /*std::cout << "(" << boost::source(e, G) << ", "
                  << boost::target(e, G) << ")\t"
                  << get(weights, e) << "\n";*/
        std::vector<double> row;
        row.push_back(source(e, G));
        row.push_back(target(e, G));
        row.push_back(get(weights, e));
        tree.push_back(row);
    }
    return tree;
}

int main(int argc, char *argv[]) {
    MaxSpanTree mst;
    std::vector< std::vector<double> > str_scores(
            {
            {0.5397449,  0.0,        0.13559636, 0.2639094,  0.04631957, 0.01442981},
            {0.17242125, 0.3721276,  0.0,        0.39170837, 0.05113386, 0.01260892},
            {0.22406709, 0.55360097, 0.16174473, 0.0,        0.04900267, 0.01158457},
            {0.04610405, 0.37811878, 0.12656094, 0.4387972,  0.0,        0.010419},
            {0.01766272, 0.37590384, 0.13813163, 0.41057062, 0.05773119, 0.0 }});

    std::vector< std::vector<double> > tree = mst.get_tree(str_scores);
    for (auto & leaf : tree){
        for (auto & element : leaf) {
            std::cout << element << ' ';
        }
    }

}

