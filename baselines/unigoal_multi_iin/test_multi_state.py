from types import SimpleNamespace

from multi_state import retarget_graph_preserving_memory


def test_retarget_preserves_scene_memory_and_resets_goal_state():
    node = SimpleNamespace(reason="old", distance=9, score=0.9,
                           exploration_level=3)
    room = SimpleNamespace(exploration_level=4, group_nodes=[object()],
                           nodes=[node])
    graph = SimpleNamespace(nodes=[node], room_nodes=[room],
                            group_nodes=[object()], objects=[object()],
                            edge_list=[object()], matcher=object(), scores=[1])
    objects, edges, nodes = graph.objects, graph.edge_list, graph.nodes

    retarget_graph_preserving_memory(graph)

    assert graph.objects is objects
    assert graph.edge_list is edges
    assert graph.nodes is nodes
    assert node.reason is None and node.distance == 2 and node.score == 0.5
    assert node.exploration_level == 0
    assert room.nodes == [node] and room.group_nodes == []
    assert graph.group_nodes == []
    assert not hasattr(graph, "matcher") and not hasattr(graph, "scores")
