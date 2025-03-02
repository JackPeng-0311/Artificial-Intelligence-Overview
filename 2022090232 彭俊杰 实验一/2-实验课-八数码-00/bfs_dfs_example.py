# how bfs and dfs search on a given graph
class Node:
    def __init__(self, state, parent):
        self.state = state
        self.parent = parent

    def __eq__(self, other):
        return self.state == other.state


graph = {
    "0": ["1", "4", "5"],  # the order of 1,4,5 affects the solution of dfs
    "1": ["0", "2"],
    "2": ["1", "3"],
    "3": ["2", "4"],
    "4": ["0", "3", "5"],
    "5": ["4", "0"],
}

start_state = "0"
end_state = "3"

open_table = []
open_table.append(Node(start_state, None))
visited = {}
path = []


def dfs():
    while len(open_table) != 0:
        cur_node = open_table[-1]
        open_table.pop()
        visited[cur_node.state] = True
        cur_state_list = graph[cur_node.state]
        for expand_state in cur_state_list:
            if expand_state not in visited:
                if expand_state == end_state:
                    tmp_node = Node(expand_state, cur_node)
                    while tmp_node is not None:
                        path.append(tmp_node.state)
                        tmp_node = tmp_node.parent
                    return True
                open_table.append(Node(expand_state, cur_node))
    return False


def bfs():
    while len(open_table) != 0:
        cur_node = open_table[0]
        open_table.pop(0)
        visited[cur_node.state] = True
        cur_state_list = graph[cur_node.state]
        for expand_state in cur_state_list:
            if expand_state not in visited:
                if expand_state == end_state:
                    tmp_node = Node(expand_state, cur_node)
                    while tmp_node is not None:
                        path.append(tmp_node.state)
                        tmp_node = tmp_node.parent
                    return True
                open_table.append(Node(expand_state, cur_node))
    return False


def print_path():
    cnt = 0
    for p in path[::-1]:
        if cnt == 0:
            cnt += 1
            print(p, end='')
        else:
            print(f'->{p}', end='')
    print('')


bfs()
print_path()
