from types import SimpleNamespace

import numpy as np

from baselines.unigoal_m.k2.state import (_stabilize_target2_obstacles,
                         retarget_graph_preserving_memory)


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


def test_translation_stabilization_changes_only_old_explored_overlap():
    reference = np.zeros((11, 64, 64), dtype=np.float32)
    reference[1, 8:56, 8:56] = 1
    reference[0, 20:22, 12:50] = 1
    target2 = reference.copy()
    # A translated duplicate in target-1 explored space must be rejected.
    target2[0, 23:25, 12:50] = 1
    # A target-2 wall in previously unknown space must remain untouched.
    target2[0, 4:6, 5:30] = 1

    corrected, report = _stabilize_target2_obstacles(target2, reference)

    assert corrected[0, 23:25, 12:50].sum() == 0
    assert corrected[0, 4:6, 5:30].sum() == 50
    assert report["removed_duplicate_obstacle_pixels"] == 76
