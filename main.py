from typing import List
import timeit
import random


# Beatriz Avanzi Ecli       RA 108612
# Sarah Anduca              RA 115506

# Conjuntos disjuntos
class DisjointSets:
    parent = {}
    rank = {}

    # Cria um subconjunto com o elemento vertice passado
    def MakeSet(self, vertex):
        self.parent[vertex] = vertex
        self.rank[vertex] = 0

    # Encontra o conjunto do elemento
    def FindSet(self, vertex):
        if self.parent[vertex] == vertex:
            return vertex
        return self.FindSet(self.parent[vertex])

    # Une dois conjuntos
    def Union(self, vertexU, vertexV):
        vertexURoot = self.FindSet(vertexU)
        vertexVRoot = self.FindSet(vertexV)

        if self.rank[vertexURoot] > self.rank[vertexVRoot]:
            self.parent[vertexVRoot] = vertexURoot
        elif self.rank[vertexURoot] < self.rank[vertexVRoot]:
            self.parent[vertexURoot] = vertexVRoot
        else:
            self.parent[vertexURoot] = vertexVRoot
            self.rank[vertexVRoot] += 1


# Classe que representa vertice
class Vertex:
    def __init__(self, num: int) -> None:
        self.d = None
        self.color = None
        self.parent = None
        self.visited = False
        self.available = True
        self.key = 0
        self.num = num
        self.adj: List[Vertex] = []


# Classe que representa aresta
class Edge:
    weight: float
    u: int
    v: int

    def __init__(self, u: int, v: int) -> None:
        self.weight = 0
        self.u = u
        self.v = v


# Classe que representa o grafo
class Graph:
    numberOfVertices: int
    numberOfEdges: int
    edgesWeight: List[float]

    def __init__(self, n: int) -> None:
        self.vertices = [Vertex(i) for i in range(n)]
        self.edges = []
        self.numberOfVertices = n
        self.numberOfEdges = 0
        self.isConnected = True
        self.edgesWeight: dict = {}

    # Adiciona uma aresta
    def addEdge(self, u: int, v: int):
        self.vertices[u].adj.append(self.vertices[v])
        self.vertices[v].adj.append(self.vertices[u])
        self.numberOfEdges = self.numberOfEdges + 1
        self.edges.append(Edge(u, v))

    # Pega o "id" de um vertice aleatorio do grafo
    def pickRandomVertexId(self) -> int:
        n: int = random.randrange(len(self.vertices))
        return n

    # Inicialização dos vertices
    def initializeAllVertex(self):
        for v in self.vertices:
            v.d = -1
            v.parent = None
            v.color = "branco"

    # BFS que retorna o ultimo vertice dequeued
    def BFS(self, s: Vertex):
        self.initializeAllVertex()
        s.d = 0
        s.color = "cinza"
        Q = [s]
        u = None

        while Q:
            u = Q.pop(0)
            for v in self.vertices[u.num].adj:
                if v.color == "branco":
                    v.color = "cinza"
                    v.d = u.d + 1
                    v.parent = u
                    Q.append(v)
            u.color = "preto"

        return u

    # Verifica se é uma arvore
    def isTree(self):
        # ve se arestas = vertices - 1
        if not self.numberOfEdges == self.numberOfVertices - 1:
            return False

        # ve se é conexo
        x: int = self.pickRandomVertexId()
        s: Vertex = self.vertices[x]

        self.BFS(s)

        self.isConnected = True

        for v in self.vertices:
            if v.color == "branco":
                self.isConnected = False

        if self.isConnected:
            return True

        return False


# Forma um grafo a partir de uma lista de arestas
def MakeGraph(edgesGroup: list) -> Graph:
    graph: Graph = Graph(len(edgesGroup) + 1)

    for (weight, edge) in edgesGroup:
        graph.addEdge(edge.u, edge.v)
        graph.edgesWeight[weight] = edge
    return graph


# Gerar um grafo denso
def GenerateFullGraph(n: int) -> Graph:
    graph: Graph = Graph(n)
    fullGraphNumberEdges = (n * (n - 1)) / 2
    verticesAvailable = list(graph.vertices)
    verticesAvailable.pop(0)

    for vertexIndex in range(graph.numberOfVertices - 1):
        for accessibleVertex in verticesAvailable:
            graph.addEdge(vertexIndex, accessibleVertex.num)
        assert len(graph.vertices[vertexIndex].adj) == n - 1
        verticesAvailable.pop(0)

    assert graph.numberOfEdges == fullGraphNumberEdges
    return graph


# Extrair o Vertex com aresta de menor peso da fila
def ExtractMin(queue: list) -> Vertex:

    lower = Vertex(1)
    lower.key = float('inf')

    for vertex in queue:
        if vertex.key < lower.key:
            lower = vertex

    lower.available = False
    queue.remove(lower)

    return lower


# Geração de arvore aleatoria pelo algoritmo de Prim
def RandomTreePrim(n: int) -> Graph:
    graph: Graph = GenerateFullGraph(n)
    edgesWeight: dict = {}

    for edge in graph.edges:
        weight = random.random()
        edge.weight = weight
        edgesWeight[(edge.u, edge.v)] = weight
        edgesWeight[(edge.v, edge.u)] = weight

    vertex = graph.pickRandomVertexId()

    return MST_Prim(graph, graph.vertices[vertex], edgesWeight)


# MST pelo algoritmo de Prim
def MST_Prim(graph: Graph, vertexS: Vertex, edgesWeight: dict):
    for vertexU in graph.vertices:
        vertexU.key = float("inf")
        vertexU.parent = None

    vertexS.key = 0
    Q = list(graph.vertices)

    tree: Graph = Graph(graph.numberOfVertices)
    assert len(Q) == graph.numberOfVertices

    while not Q == []:
        vertexU: Vertex = ExtractMin(Q)

        if vertexU.parent:
            tree.addEdge(vertexU.parent.num, vertexU.num)
        for vertexV in graph.vertices[vertexU.num].adj:
            if vertexV.key > edgesWeight[(vertexU.num, vertexV.num)] and vertexV.available: # peso da chave atual é maior do que a disponivel em um adjacente
                vertexV.key = edgesWeight[(vertexU.num, vertexV.num)]
                vertexV.parent = vertexU

    assert tree.isTree()
    return tree


# Geração de arvore aleatoria pelo algoritmo de Kruskal
def RandomTreeKruskal(n: int) -> Graph:
    graph: Graph = GenerateFullGraph(n)
    edgesWeight: dict = {}

    for edge in graph.edges:
        weight = random.random()
        edge.weight = weight
        edgesWeight[edge] = weight

    return MST_Kruskal(graph, edgesWeight)


# MST pelo algoritmo de Kruskal
def MST_Kruskal(graph: Graph, edgesWeight: dict):
    treeGroup = []
    disjointSets = DisjointSets()

    for vertex in range(graph.numberOfVertices):
        disjointSets.MakeSet(vertex)

    edgesSorted = sorted(edgesWeight.items(), key=lambda item: item[1])

    for (edge, weight) in edgesSorted:
        if disjointSets.FindSet(edge.u) != disjointSets.FindSet(edge.v):
            treeGroup.append({weight, edge})
            disjointSets.Union(edge.u, edge.v)

    treeGroup = MakeGraph(treeGroup)
    assert treeGroup.isTree()
    return treeGroup


# Geração de arvore aleatoria de forma randomica
def RandomTreeRandomWalk(n: int) -> Graph:
    graph: Graph = Graph(n)
    for u in graph.vertices:
        u.visited = False

    x: int = graph.pickRandomVertexId()
    u: Vertex = graph.vertices[x]
    u.visited = True
    while graph.numberOfEdges < n - 1:
        y: int = graph.pickRandomVertexId()
        v: Vertex = graph.vertices[y]
        if not v.visited:
            graph.addEdge(u.num, v.num)
            v.visited = True
        u = v

    assert graph.isTree()
    return graph


# Calculo do diameter
def Diameter(tree: Graph) -> float:
    n: int = tree.pickRandomVertexId()  # pega um numero aleatorio no range da quantidade de vertices
    s: Vertex = tree.vertices[n]

    a: Vertex = tree.BFS(s)
    b: Vertex = tree.BFS(a)
    return b.d


def GenerateTXT(fileName: str, elements):
    try:
        file = open(fileName, 'r+')
    except FileNotFoundError:
        file = open(fileName, 'w+')

    for sublist in elements:
        file.write(str(sublist[0]) + " " + str(sublist[1]) + "\n")

    file.close()


def TEST_Diameter():
    graph1: Graph = Graph(4)
    graph1.addEdge(0, 1)
    graph1.addEdge(2, 1)
    graph1.addEdge(3, 1)
    assert Diameter(graph1) == 2

    graph2: Graph = Graph(9)
    graph2.addEdge(0, 1)
    graph2.addEdge(0, 2)
    graph2.addEdge(2, 3)
    graph2.addEdge(2, 8)
    graph2.addEdge(3, 6)
    graph2.addEdge(3, 4)
    graph2.addEdge(4, 5)
    graph2.addEdge(5, 7)
    assert Diameter(graph2) == 6


def TEST_isTree():
    # caso 1 - não é arvore, pois arestas != vertices - 1

    graph1: Graph = Graph(5)
    graph1.addEdge(0, 1)
    graph1.addEdge(2, 1)
    graph1.addEdge(3, 1)
    assert not graph1.isTree()

    # caso 2 - não é arvore, pois não é conexo
    graph2: Graph = Graph(5)
    graph2.addEdge(0, 1)
    graph2.addEdge(0, 2)
    graph2.addEdge(1, 3)
    graph2.addEdge(2, 1)
    assert not graph2.isTree()

    # caso 3 - é arvore
    graph3: Graph = Graph(5)
    graph3.addEdge(0, 1)
    graph3.addEdge(0, 2)
    graph3.addEdge(1, 3)
    graph3.addEdge(2, 4)
    assert graph3.isTree()


def TEST_ExtractMin():
    # caso 1 - retorna key 1
    graph1: Graph = Graph(5)
    graph1.vertices[0].key = 10
    graph1.vertices[1].key = 50
    graph1.vertices[2].key = 1
    graph1.vertices[3].key = 2
    graph1.vertices[4].key = 3

    assert ExtractMin(graph1.vertices).key == 1

    # caso 2 - retorna key 0
    graph2: Graph = Graph(10)
    graph2.vertices[0].key = 500
    graph2.vertices[1].key = 50
    graph2.vertices[2].key = 30
    graph2.vertices[3].key = 2
    graph2.vertices[4].key = 1

    assert ExtractMin(graph2.vertices).key == 0

    # caso 3 - retorna key 0
    for n in range(500):
        graph3: Graph = Graph(50)
        for i in range(50):
            graph3.vertices[i].key = float("inf")

        graph3.vertices[20].key = 0
        assert ExtractMin(graph3.vertices).key == 0


def RunAllTests():
    TEST_Diameter()
    TEST_isTree()
    TEST_ExtractMin()


def RunRandomWalk():
    random_walk_result = []
    entries = [250, 500, 750, 1000, 1250, 1500, 1750, 2000]
    for entry in entries:
        diameterSum = 0
        n = 500
        for i in range(n):
            tree = RandomTreeRandomWalk(entry)
            diameter = Diameter(tree)
            diameterSum = diameter + diameterSum

        diameterAvg = diameterSum / n
        random_walk_result.append([entry, diameterAvg])

    GenerateTXT("randomwalk.txt", random_walk_result)


def RunKrukalMST():
    result = []
    entries = [250, 500, 750, 1000, 1250, 1500, 1750, 2000]
    for entry in entries:
        diameterSum = 0
        n = 500
        for i in range(n):
            tree = RandomTreeKruskal(entry)
            diameter = Diameter(tree)
            diameterSum = diameter + diameterSum

        diameterAvg = diameterSum / n
        result.append([entry, diameterAvg])
    GenerateTXT("kruskal.txt", result)


def RunPrimMST():
    result = []
    entries = [250, 500, 750, 1000, 1250, 1500, 1750, 2000]
    for entry in entries:
        diameterSum = 0
        n = 500
        for i in range(n):
            tree = RandomTreePrim(entry)
            diameter = Diameter(tree)
            diameterSum = diameter + diameterSum

        diameterAvg = diameterSum / n
        result.append([entry, diameterAvg])
    GenerateTXT("prim.txt", result)


def main():
    inicio = timeit.default_timer()

    RunAllTests()
    fim = timeit.default_timer()
    print('Cálculo de duração All Tests: %f' % (fim - inicio))
    RunRandomWalk()
    fim = timeit.default_timer()
    print('Cálculo de duração Random Walk: %f' % (fim - inicio))
   
    RunKrukalMST()
    fim = timeit.default_timer()
    print('Cálculo de duração Kruskal: %f' % (fim - inicio))

    RunPrimMST()
    fim = timeit.default_timer()
    print('Cálculo de duração Prim: %f' % (fim - inicio))
    


    # mostra o tempo que demorou para execução
    print('Cálculo de duração: %f' % (fim - inicio))


if __name__ == '__main__':
    main()