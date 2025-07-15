#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

class Node
{

public:

   Node(int idx, std::map<int, real_t> & queue) : 
      _idx{idx}, 
      _prev_node{-1}, 
      _queue{queue}
   {}

   ~Node() = default;

   void updateNeighbours();

   void setPreviousNode(int n){ _prev_node = n; }
   int getPreviousNode(){ return _prev_node; }
   real_t getDistance() { return _distance; }

   void addEdge(std::shared_ptr<Node> neigh, real_t dist);
   void resetDistanceIfShorter(real_t d, int origin_idx);
    
private:

   int _idx;

   // Dijkstra's algorithm doesn't allow negative distances, 
   // so here we are using -1 as a placeholder for "infinite" distance
   real_t _distance = -1;
   int _prev_node;
   std::vector<std::shared_ptr<Node>> _neighbours;
   std::vector<real_t> _weights;
   std::map<int, real_t> & _queue;
    
};

class Graph
{

public:

   void setSource(int s);
   void setDestination(int d);
   void addNode(int idx);
   void addEdge(int from, int to, real_t dist);
   void dijkstra();
   std::map<int, real_t> & getQueue() { return _queue; }
   void getShortestPath();
   void printShortestPath();
   void printShortestPathInParaviewForm();
   int getClosestNode();
   void exportCSVPath(std::string filename);

private:

   std::map<int, real_t> _queue;
   std::unordered_map<int, std::shared_ptr<Node>> _nodes;
   std::vector<int> _path;
   real_t _path_dist = 0;
   int _source;
   int _dest;

};

void Node::resetDistanceIfShorter(real_t d, int origin_idx)
{ 
   if (_distance < 0 || d < _distance)
   {
      _distance = d;
      _prev_node = origin_idx;

      // If the node is already in the queue, replace it
      const auto it = _queue.find(_idx);
      if (it != _queue.end())
         _queue.erase(it);
      _queue[_idx] = _distance;
   }
}

void Node::addEdge(std::shared_ptr<Node> neigh, real_t dist)
{
   _neighbours.push_back(neigh);
   _weights.push_back(dist);
}

void Node::updateNeighbours()
{
   _queue.erase(_idx);

   for (int i=0; i<_neighbours.size(); ++i)
      _neighbours[i]->resetDistanceIfShorter(_distance+_weights[i], _idx);
}

void Graph::setSource(int s)
{
   if (auto search = _nodes.find(s); search == _nodes.end())
   {
      std::cout << "Node chosen as source has not been found in the mesh. Please select a valid node" << std::endl;
      assert(false);
   }
   _source = s; 
}

void Graph::setDestination(int d)
{
   if (auto search = _nodes.find(d); search == _nodes.end())
   {
      std::cout << "Node chosen as destination has not been found in the mesh. Please select a valid node" << std::endl;
      assert(false);
   }
   _dest = d;
}

void Graph::dijkstra()
{
   assert(_nodes.size() > 0);
   _nodes[_source]->resetDistanceIfShorter(0,0);
   int current_node = _source;

   while (current_node != _dest)
   {
      _nodes[current_node]->updateNeighbours();
      assert(_queue.size());
      current_node = getClosestNode();
   }

    getShortestPath();
}

int Graph::getClosestNode()
{
    return _queue.begin()->first;
}

void Graph::addEdge(int from, int to, real_t dist)
{
   if (auto search = _nodes.find(from); search == _nodes.end())
     addNode(from);
   if (auto search = _nodes.find(to); search == _nodes.end())
     addNode(to);

   _nodes[from]->addEdge(_nodes[to], dist);
   _nodes[to]->addEdge(_nodes[from], dist); // Assuming undirected graph
}

void Graph::addNode(int idx)
{
   _nodes.insert({idx, std::make_shared<Node>(idx,_queue)});
}

void Graph::getShortestPath()
{
    std::vector<int> temp_path;
    int prev_node = _dest;
    while (prev_node != _source)
    {
        temp_path.push_back(prev_node);
        prev_node = _nodes[prev_node]->getPreviousNode();
    }

    _path.clear();
    _path.push_back(_source);
    for (int i=temp_path.size()-1; i>=0; --i)
        _path.push_back(temp_path[i]);

    _path_dist = _nodes[_dest]->getDistance();
}

void Graph::printShortestPath()
{
   std::cout << "Shortest path from " << _source << " to " << _dest 
              << " is: " << std::endl;

   std::cout << "[";
   for (int i=0; i<_path.size(); ++i)
        std::cout << _path[i] << (i < _path.size()-1 ? ", " : "");
   std::cout << "]" << std::endl;

    std::cout << std::endl << " with distance: " << _path_dist << std::endl;
}

void getEdgeTableWithoutDuplicates(Mesh & mesh,std::vector<std::vector<int>> & etable, std::vector<int> & map_to_duplicate)
{
   etable.clear();
   Table * evtable_dup = mesh.GetEdgeVertexTable();
   map_to_duplicate.clear();
   map_to_duplicate.resize(mesh.GetNV());
   std::unordered_map<std::string,int> unique_nodes;

   for (int i=0; i<mesh.GetNV(); ++i)
   {
      std::string ikey{""};
      for (int d=0; d<mesh.Dimension(); ++d) 
         ikey += std::to_string(mesh.GetVertex(i)[d]);

      if(auto search = unique_nodes.find(ikey); search != unique_nodes.end())
      {
         map_to_duplicate[i] = search->second;
      }
      else
      {
         unique_nodes.insert({ikey,i});
         map_to_duplicate[i] = i;
      }
   }

   for (int r=0; r<evtable_dup->Size(); ++r)
   {
      Array<int> row;
      evtable_dup->GetRow(r, row);
      std::vector<int> edge;
      edge.push_back(map_to_duplicate[row[0]]);
      edge.push_back(map_to_duplicate[row[1]]);
      etable.push_back(edge);
   }

   std::cout << "Mesh has " << unique_nodes.size() << " effective vertices and " 
      << mesh.GetNV()-unique_nodes.size() << " duplicates\n" << std::endl;
}

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "coil_submesh.mesh";
   int source = 0;
   int dest = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&source, "-s", "--source",
                 "Source node for the pathfinding algorithm.");
   args.AddOption(&dest, "-d", "--dest",
                 "Destination node for the pathfinding algorithm.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);
   std::cout << std::endl;

   Mesh mesh(mesh_file, 0, 0);
   int dim = mesh.Dimension();

   std::cout << "Mesh has " << mesh.GetNV() << " vertices" << std::endl;
   std::cout << "Mesh has " << mesh.GetNEdges() << " edges" << std::endl;

   MFEM_ASSERT(mesh.bdr_attributes.Size(),
               "This example does not support meshes"
               " without boundary attributes."
              )

   std::vector<std::vector<int>> edge_table;
   std::vector<int> map_to_duplicate;
   getEdgeTableWithoutDuplicates(mesh, edge_table, map_to_duplicate);

   Graph g;

   for (int i=0; i<edge_table.size(); ++i)
   {
      int v1 = edge_table[i][0];
      int v2 = edge_table[i][1];
      real_t dist = Distance(mesh.GetVertex(v1), mesh.GetVertex(v2), dim);
      g.addEdge(v1, v2, dist);
   }

   g.setSource(map_to_duplicate[source]);
   g.setDestination(map_to_duplicate[dest]);

   g.dijkstra();
   g.printShortestPath();

}