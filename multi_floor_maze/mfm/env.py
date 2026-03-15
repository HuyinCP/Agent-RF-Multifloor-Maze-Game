"""
Multi-Floor Maze Environment - Optimized for local training.
Key optimization: NO BFS in step(). Distance cached at reset & key/door events.
"""
import copy
import heapq
import random
from collections import defaultdict, deque
from enum import IntEnum

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# ═══════════════════════════════════════════════════════════════
# Tiles & Constants
# ═══════════════════════════════════════════════════════════════

class Tile(IntEnum):
    FLOOR = 0
    WALL = 1
    STAIR_UP = 3
    STAIR_DOWN = 4
    GOAL = 5
    TRAP = 6
    START = 7
    DOOR = 10
    KEY = 20
    HEALTH_PACK = 30
    AMMO_PACK = 31


DIR_DELTA = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}


# ═══════════════════════════════════════════════════════════════
# BSP Map Generator
# ═══════════════════════════════════════════════════════════════

class _BSPNode:
    __slots__ = ("x", "y", "w", "h", "left", "right", "room")

    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.left = self.right = self.room = None

    def leaves(self):
        if not self.left and not self.right:
            return [self]
        out = []
        if self.left:
            out.extend(self.left.leaves())
        if self.right:
            out.extend(self.right.leaves())
        return out


class ProceduralMapGenerator:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(
            seed if seed is not None else None
        )

    def _bsp_params(self, w, h):
        s = min(w, h)
        if s <= 9:
            return 3, 2
        if s <= 14:
            return 4, 3
        return 5, 4

    def _bsp_split(self, node, ml, depth=0, md=4):
        if depth >= md:
            return
        can_h = node.w >= ml * 2
        can_v = node.h >= ml * 2
        if not can_h and not can_v:
            return
        if can_h and can_v:
            horiz = self.rng.random() < 0.5
        else:
            horiz = can_h
        if horiz:
            sp = self.rng.randint(ml, node.w - ml)
            node.left = _BSPNode(node.x, node.y, sp, node.h)
            node.right = _BSPNode(node.x + sp, node.y, node.w - sp, node.h)
        else:
            sp = self.rng.randint(ml, node.h - ml)
            node.left = _BSPNode(node.x, node.y, node.w, sp)
            node.right = _BSPNode(node.x, node.y + sp, node.w, node.h - sp)
        self._bsp_split(node.left, ml, depth + 1, md)
        self._bsp_split(node.right, ml, depth + 1, md)

    def _carve_rooms(self, node):
        for leaf in node.leaves():
            min_rw = min(3, leaf.w)
            min_rh = min(3, leaf.h)
            max_rw = max(min_rw, leaf.w - 2)
            max_rh = max(min_rh, leaf.h - 2)
            rw = self.rng.randint(min_rw, max_rw)
            rh = self.rng.randint(min_rh, max_rh)
            rx = self.rng.randint(leaf.x, max(leaf.x, leaf.x + leaf.w - rw))
            ry = self.rng.randint(leaf.y, max(leaf.y, leaf.y + leaf.h - rh))
            leaf.room = (rx, ry, rw, rh)

    def _get_rooms(self, node):
        return [leaf.room for leaf in node.leaves() if leaf.room]

    def _sibling_edges(self, node, rooms):
        edges = []
        if node.left and node.right:
            lr = self._get_rooms(node.left)
            rr = self._get_rooms(node.right)
            if lr and rr:
                bd, bp = float("inf"), None
                for a in lr:
                    ac = (a[0] + a[2] // 2, a[1] + a[3] // 2)
                    for b in rr:
                        bc = (b[0] + b[2] // 2, b[1] + b[3] // 2)
                        d = abs(ac[0] - bc[0]) + abs(ac[1] - bc[1])
                        if d < bd:
                            bd, bp = d, (rooms.index(a), rooms.index(b))
                if bp:
                    edges.append(bp)
            edges.extend(self._sibling_edges(node.left, rooms))
            edges.extend(self._sibling_edges(node.right, rooms))
        return edges

    def _build_graph(self, root, rooms, add_loops=True):
        n = len(rooms)
        if n <= 1:
            return defaultdict(set)
        adj = defaultdict(set)
        for a, b in self._sibling_edges(root, rooms):
            adj[a].add(b)
            adj[b].add(a)
        visited = {0}
        q = deque([0])
        while q:
            c = q.popleft()
            for nb in adj[c]:
                if nb not in visited:
                    visited.add(nb)
                    q.append(nb)
        for i in range(n):
            if i not in visited:
                bd, bj = float("inf"), 0
                ic = (rooms[i][0] + rooms[i][2] // 2, rooms[i][1] + rooms[i][3] // 2)
                for j in visited:
                    jc = (
                        rooms[j][0] + rooms[j][2] // 2,
                        rooms[j][1] + rooms[j][3] // 2,
                    )
                    d = abs(ic[0] - jc[0]) + abs(ic[1] - jc[1])
                    if d < bd:
                        bd, bj = d, j
                adj[i].add(bj)
                adj[bj].add(i)
                visited.add(i)
                q2 = deque([i])
                while q2:
                    c2 = q2.popleft()
                    for nb in adj[c2]:
                        if nb not in visited:
                            visited.add(nb)
                            q2.append(nb)
        if add_loops:
            extra = []
            for i in range(n):
                for j in range(i + 1, n):
                    if j not in adj[i]:
                        ci = (
                            rooms[i][0] + rooms[i][2] // 2,
                            rooms[i][1] + rooms[i][3] // 2,
                        )
                        cj = (
                            rooms[j][0] + rooms[j][2] // 2,
                            rooms[j][1] + rooms[j][3] // 2,
                        )
                        extra.append(
                            (abs(ci[0] - cj[0]) + abs(ci[1] - cj[1]), i, j)
                        )
            extra.sort()
            pool = extra[: max(1, len(extra) // 3)]
            n_add = min(max(1, n // 4), len(pool))
            if pool:
                for _, i, j in self.rng.sample(pool, n_add):
                    adj[i].add(j)
                    adj[j].add(i)
        return adj

    def _carve_corridors(self, grid, rooms, graph):
        done = set()
        for i in graph:
            for j in graph[i]:
                e = (min(i, j), max(i, j))
                if e in done:
                    continue
                done.add(e)
                self._carve_one(grid, rooms[i], rooms[j])

    def _carve_one(self, grid, r1, r2):
        c1x, c1y = r1[0] + r1[2] // 2, r1[1] + r1[3] // 2
        c2x, c2y = r2[0] + r2[2] // 2, r2[1] + r2[3] // 2
        h, w = grid.shape
        if self.rng.random() < 0.5:
            self._hc(grid, c1x, c2x, c1y, w, h)
            self._vc(grid, c1y, c2y, c2x, w, h)
        else:
            self._vc(grid, c1y, c2y, c1x, w, h)
            self._hc(grid, c1x, c2x, c2y, w, h)

    def _hc(self, g, x1, x2, y, w, h):
        for x in range(min(x1, x2), max(x1, x2) + 1):
            if 0 < y < h - 1 and 0 < x < w - 1:
                g[y][x] = Tile.FLOOR

    def _vc(self, g, y1, y2, x, w, h):
        for y in range(min(y1, y2), max(y1, y2) + 1):
            if 0 < y < h - 1 and 0 < x < w - 1:
                g[y][x] = Tile.FLOOR

    def _critical_path(self, graph, start, goal, n):
        if start == goal:
            return [start]
        vis = {start}
        q = deque([(start, [start])])
        while q:
            cur, path = q.popleft()
            if cur == goal:
                return path
            for nb in sorted(graph[cur]):
                if nb not in vis:
                    vis.add(nb)
                    q.append((nb, path + [nb]))
        return list(range(n))

    def _assign_zones(self, rooms, graph, cpath, n_enemies):
        n = len(rooms)
        zones = {}
        cp_set = set(cpath)
        zones[cpath[0]] = "spawn"
        zones[cpath[-1]] = "exit"
        for i in range(n):
            if i not in cp_set:
                zones[i] = "treasure"
        placed = 0
        if len(cpath) > 3:
            mid = len(cpath) // 2
            for idx in range(max(1, mid - 1), min(len(cpath) - 1, mid + 2)):
                ri = cpath[idx]
                if ri not in zones and placed < max(1, n_enemies):
                    zones[ri] = "combat"
                    placed += 1
        for ri in cpath:
            if ri not in zones:
                zones[ri] = "passage"
        return zones

    def _pick_in_room(self, grid, room, used):
        rx, ry, rw, rh = room
        h, w = grid.shape
        cands = [
            (x, y)
            for y in range(max(1, ry), min(h - 1, ry + rh))
            for x in range(max(1, rx), min(w - 1, rx + rw))
            if grid[y][x] == Tile.FLOOR and (x, y) not in used
        ]
        if cands:
            c = self.rng.choice(cands)
            used.add(c)
            return c
        return self._pick_floor(grid, used)

    def _pick_floor(self, grid, exc=None):
        h, w = grid.shape
        ex = exc or set()
        tiles = [
            (x, y)
            for y in range(h)
            for x in range(w)
            if grid[y][x] == Tile.FLOOR and (x, y) not in ex
        ]
        if tiles:
            c = self.rng.choice(tiles)
            if exc is not None:
                exc.add(c)
            return c
        return (1, 1)

    def _collect_candidates(self, grid, rooms, zones, zone_names, used):
        """Danh sách ô FLOOR chưa dùng trong các room thuộc zone_names (đặt từng ô có kiểm soát)."""
        out = []
        for ri, r in enumerate(rooms):
            if zones.get(ri) not in zone_names:
                continue
            rx, ry, rw, rh = r
            h, w = grid.shape
            for y in range(max(1, ry), min(h - 1, ry + rh)):
                for x in range(max(1, rx), min(w - 1, rx + rw)):
                    if grid[y][x] == Tile.FLOOR and (x, y) not in used:
                        out.append((x, y))
        return out

    def _in_any_room(self, x, y, rooms):
        for r in rooms:
            if r[0] <= x < r[0] + r[2] and r[1] <= y < r[1] + r[3]:
                return True
        return False

    def _bfs_tile_path(self, grid, sx, sy, gx, gy):
        h, w = grid.shape
        parent = {(sx, sy): None}
        q = deque([(sx, sy)])
        while q:
            cx, cy = q.popleft()
            if cx == gx and cy == gy:
                path = []
                cur = (cx, cy)
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if (
                    0 <= nx < w
                    and 0 <= ny < h
                    and (nx, ny) not in parent
                    and int(grid[ny][nx]) != int(Tile.WALL)
                ):
                    parent[(nx, ny)] = (cx, cy)
                    q.append((nx, ny))
        return None

    def _bfs_tile_connected(self, grid, sx, sy, gx, gy, extra_block=None):
        h, w = grid.shape
        block = {int(Tile.WALL)}
        if extra_block:
            block = block | extra_block
        vis = {(sx, sy)}
        q = deque([(sx, sy)])
        while q:
            cx, cy = q.popleft()
            if cx == gx and cy == gy:
                return True
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if (
                    0 <= nx < w
                    and 0 <= ny < h
                    and (nx, ny) not in vis
                    and int(grid[ny][nx]) not in block
                ):
                    vis.add((nx, ny))
                    q.append((nx, ny))
        return False

    def _bfs_reachable_set(self, floors, start, block_types):
        sx, sy, sf = start
        nf = len(floors)
        h, w = floors[0].shape
        su, sd = {}, {}
        for f in range(nf):
            for y in range(h):
                for x in range(w):
                    t = int(floors[f][y][x])
                    if t == int(Tile.STAIR_UP) and f + 1 < nf:
                        su[(x, y, f)] = (x, y, f + 1)
                    elif t == int(Tile.STAIR_DOWN) and f > 0:
                        sd[(x, y, f)] = (x, y, f - 1)
        vis = {(sx, sy, sf)}
        q = deque([(sx, sy, sf)])
        while q:
            cx, cy, cf = q.popleft()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if (
                    0 <= nx < w
                    and 0 <= ny < h
                    and (nx, ny, cf) not in vis
                ):
                    t = int(floors[cf][ny][nx])
                    if t != int(Tile.WALL) and t not in block_types:
                        vis.add((nx, ny, cf))
                        q.append((nx, ny, cf))
            for stair_dict in (su, sd):
                if (cx, cy, cf) in stair_dict:
                    dest = stair_dict[(cx, cy, cf)]
                    if dest not in vis:
                        vis.add(dest)
                        q.append(dest)
        return vis

    # Giới hạn số ô thử khi tìm choke (sinh nhanh hơn, logic tương đương)
    _MAX_CHOKE_CANDIDATES = 14

    def _find_single_tile_choke(self, grid, rooms, entry_x, entry_y, exit_x, exit_y, used):
        """Find a single corridor tile that disconnects entry from exit when blocked."""
        h, w = grid.shape
        block = {int(Tile.WALL), int(Tile.DOOR)}
        parent = {(entry_x, entry_y): None}
        q = deque([(entry_x, entry_y)])
        while q:
            cx, cy = q.popleft()
            if cx == exit_x and cy == exit_y:
                break
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if (
                    0 <= nx < w
                    and 0 <= ny < h
                    and (nx, ny) not in parent
                    and int(grid[ny][nx]) not in block
                ):
                    parent[(nx, ny)] = (cx, cy)
                    q.append((nx, ny))
        if (exit_x, exit_y) not in parent:
            return None
        path = []
        cur = (exit_x, exit_y)
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        mid = path[1:-1]
        for cx, cy in mid[: self._MAX_CHOKE_CANDIDATES]:
            if (cx, cy) in used or self._in_any_room(cx, cy, rooms):
                continue
            old = int(grid[cy][cx])
            grid[cy][cx] = int(Tile.WALL)
            connected = self._bfs_tile_connected(grid, entry_x, entry_y, exit_x, exit_y)
            grid[cy][cx] = old
            if not connected:
                return (cx, cy)
        return None

    def _perpendicular_seal(self, grid, cx, cy, rooms, used):
        h, w = grid.shape
        best = None
        for axis in ["v", "h"]:
            seal = [(cx, cy)]
            if axis == "v":
                for d in [-1, 1]:
                    yy = cy + d
                    while (
                        0 < yy < h - 1
                        and int(grid[yy][cx]) == int(Tile.FLOOR)
                        and not self._in_any_room(cx, yy, rooms)
                    ):
                        seal.append((cx, yy))
                        yy += d
            else:
                for d in [-1, 1]:
                    xx = cx + d
                    while (
                        0 < xx < w - 1
                        and int(grid[cy][xx]) == int(Tile.FLOOR)
                        and not self._in_any_room(xx, cy, rooms)
                    ):
                        seal.append((xx, cy))
                        xx += d
            seal = [(x, y) for x, y in seal if (x, y) not in used]
            if seal and len(seal) <= 2:
                if best is None or len(seal) < len(best):
                    best = seal
        return best

    def _find_stair_toward(self, grid, df, target_floor, h, w):
        look_for = (
            int(Tile.STAIR_UP) if target_floor > df else int(Tile.STAIR_DOWN)
        )
        for yy in range(h):
            for xx in range(w):
                if int(grid[yy][xx]) == look_for:
                    return (xx, yy)
        for yy in range(h):
            for xx in range(w):
                if int(grid[yy][xx]) in (
                    int(Tile.STAIR_DOWN),
                    int(Tile.STAIR_UP),
                ):
                    return (xx, yy)
        return None

    def _place_lock(self, all_f, all_r, all_g, kf, df, start, goal, used_per_floor):
        """Đặt 1 cửa trên floor df và 1 chìa trên floor kf. used_per_floor: list of set (x,y) per floor."""
        grid = all_f[df]
        rooms = all_r[df]
        used_df = used_per_floor[df]
        used_kf = used_per_floor[kf]
        h, w = grid.shape
        sx, sy, sfr = start
        gx, gy, gfr = goal

        entry_x, entry_y = sx, sy
        if df != sfr:
            sp = self._find_stair_toward(grid, df, sfr, h, w)
            if sp:
                entry_x, entry_y = sp
            else:
                return False

        exit_x, exit_y = gx, gy
        if df != gfr:
            sp = self._find_stair_toward(grid, df, gfr, h, w)
            if sp:
                exit_x, exit_y = sp
            else:
                return False

        if entry_x == exit_x and entry_y == exit_y:
            return False

        tile_path = self._bfs_tile_path(
            grid, entry_x, entry_y, exit_x, exit_y
        )
        if not tile_path or len(tile_path) < 3:
            return False

        door_type = int(Tile.DOOR)
        key_type = int(Tile.KEY)

        choke = self._find_single_tile_choke(
            grid, rooms, entry_x, entry_y, exit_x, exit_y, used_df
        )
        if choke:
            cx, cy = choke
            grid[cy][cx] = door_type
            reachable = self._bfs_reachable_set(all_f, start, {door_type})
            key_placed = False
            krooms = list(all_r[kf])
            self.rng.shuffle(krooms)
            for kr in krooms:
                kcx, kcy = kr[0] + kr[2] // 2, kr[1] + kr[3] // 2
                if (kcx, kcy, kf) in reachable:
                    kt = self._pick_in_room(all_f[kf], kr, used_kf)
                    if (kt[0], kt[1], kf) in reachable:
                        all_f[kf][kt[1]][kt[0]] = key_type
                        used_kf.add(kt)
                        key_placed = True
                        break
            if not key_placed:
                kh, kw = all_f[kf].shape
                for y2 in range(1, kh - 1):
                    for x2 in range(1, kw - 1):
                        if (
                            (x2, y2, kf) in reachable
                            and int(all_f[kf][y2][x2]) == int(Tile.FLOOR)
                            and (x2, y2) not in used_kf
                        ):
                            all_f[kf][y2][x2] = key_type
                            used_kf.add((x2, y2))
                            key_placed = True
                            break
                    if key_placed:
                        break
            if key_placed:
                used_df.add((cx, cy))
                return True
            grid[cy][cx] = Tile.FLOOR

        for cx, cy in (tile_path[1:-1])[: self._MAX_CHOKE_CANDIDATES]:
            seal = self._perpendicular_seal(grid, cx, cy, rooms, used_df)
            if seal and len(seal) == 1:
                old = {(x, y): int(grid[y][x]) for x, y in seal}
                for x, y in seal:
                    grid[y][x] = door_type

                if self._bfs_tile_connected(
                    grid, entry_x, entry_y, exit_x, exit_y, {door_type}
                ):
                    for (x, y), t in old.items():
                        grid[y][x] = t
                    continue

                reachable = self._bfs_reachable_set(
                    all_f, start, {door_type}
                )
                key_placed = False
                krooms = list(all_r[kf])
                self.rng.shuffle(krooms)
                for kr in krooms:
                    kcx, kcy = kr[0] + kr[2] // 2, kr[1] + kr[3] // 2
                    if (kcx, kcy, kf) in reachable:
                        kt = self._pick_in_room(all_f[kf], kr, used_kf)
                        if (kt[0], kt[1], kf) in reachable:
                            all_f[kf][kt[1]][kt[0]] = key_type
                            used_kf.add(kt)
                            key_placed = True
                            break

                if not key_placed:
                    kh, kw = all_f[kf].shape
                    for y2 in range(1, kh - 1):
                        for x2 in range(1, kw - 1):
                            if (
                                (x2, y2, kf) in reachable
                                and int(all_f[kf][y2][x2]) == int(Tile.FLOOR)
                                and (x2, y2) not in used_kf
                            ):
                                all_f[kf][y2][x2] = key_type
                                used_kf.add((x2, y2))
                                key_placed = True
                                break
                        if key_placed:
                            break

                if not key_placed:
                    for (x, y), t in old.items():
                        grid[y][x] = t
                    continue

                for x, y in seal:
                    used_df.add((x, y))
                return True

        return False

    def _bfs_multi(self, floors, start, goal):
        sx, sy, sf = start
        gx, gy, gf = goal
        nf = len(floors)
        h, w = floors[0].shape
        su, sd = {}, {}
        for f in range(nf):
            for y in range(h):
                for x in range(w):
                    if floors[f][y][x] == Tile.STAIR_UP and f + 1 < nf:
                        su[(x, y, f)] = (x, y, f + 1)
                    elif floors[f][y][x] == Tile.STAIR_DOWN and f > 0:
                        sd[(x, y, f)] = (x, y, f - 1)
        vis = set()
        q = deque([(sx, sy, sf, 0)])
        vis.add((sx, sy, sf))
        while q:
            cx, cy, cf, cd = q.popleft()
            if cx == gx and cy == gy and cf == gf:
                return cd
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if (
                    0 <= nx < w
                    and 0 <= ny < h
                    and (nx, ny, cf) not in vis
                    and int(floors[cf][ny][nx]) != Tile.WALL
                ):
                    vis.add((nx, ny, cf))
                    q.append((nx, ny, cf, cd + 1))
            for sd_ in (su, sd):
                if (cx, cy, cf) in sd_:
                    dest = sd_[(cx, cy, cf)]
                    if dest not in vis:
                        vis.add(dest)
                        q.append((*dest, cd + 1))
        return -1

    def validate(self, map_data, config):
        floors = map_data["floors"]
        nf = map_data["n_floors"]
        w, h = map_data["width"], map_data["height"]
        sx, sy, sf = map_data["start"]
        gx, gy, gf = map_data["goal"]
        for f in range(nf):
            if np.sum(floors[f] != Tile.WALL) / (w * h) < 0.20:
                return False, "low_floor_ratio"
        dist = self._bfs_multi(floors, (sx, sy, sf), (gx, gy, gf))
        if dist < 0:
            return False, "unreachable"
        if dist < max(w, h) // 2:
            return False, "too_short"
        if config.get("locks"):
            # 1) Với tất cả cửa bị chặn, mọi chìa khoá phải vẫn đi tới được
            reachable = self._bfs_reachable_set(
                floors, (sx, sy, sf), {int(Tile.DOOR)}
            )
            for f in range(nf):
                for y in range(h):
                    for x in range(w):
                        if int(floors[f][y][x]) == int(Tile.KEY) and (
                            x,
                            y,
                            f,
                        ) not in reachable:
                            return False, "key_behind_door"
            # 2) Các floor có cửa: muốn tới stair/goal trên floor đó thì BẮT BUỘC phải qua ít
            #    nhất một cửa. Nghĩa là nếu xem mọi DOOR như tường, start không được chạm tới
            #    bất kỳ ô stair/goal nào trên những floor có lock.
            critical_targets = set()
            door_floors = {lock.get("d", 0) for lock in config.get("locks", [])}
            for f in door_floors:
                grid = floors[f]
                for y in range(h):
                    for x in range(w):
                        t = int(grid[y][x])
                        if t in (int(Tile.STAIR_UP), int(Tile.STAIR_DOWN)):
                            critical_targets.add((x, y, f))
                        if f == gf and t == int(Tile.GOAL):
                            critical_targets.add((x, y, f))
            if any(t in reachable for t in critical_targets):
                return False, "door_not_blocking_critical"
        return True, "ok"

    def _gen_floor(self, w, h, add_loops=True):
        grid = np.full((h, w), Tile.WALL, dtype=np.int32)
        ml, md = self._bsp_params(w, h)
        root = _BSPNode(1, 1, w - 2, h - 2)
        self._bsp_split(root, ml, 0, md)
        self._carve_rooms(root)
        rooms = self._get_rooms(root)
        if len(rooms) < 2:
            rw1 = max(3, (w - 2) // 3)
            rh1 = max(3, (h - 2) // 3)
            rw2 = max(3, (w - 2) // 3)
            rh2 = max(3, (h - 2) // 3)
            rooms = [
                (1, 1, rw1, rh1),
                (w - rw2 - 1, h - rh2 - 1, rw2, rh2),
            ]
        for rx, ry, rw, rh in rooms:
            for y in range(ry, min(ry + rh, h - 1)):
                for x in range(rx, min(rx + rw, w - 1)):
                    if 0 < y < h - 1 and 0 < x < w - 1:
                        grid[y][x] = Tile.FLOOR
        graph = self._build_graph(root, rooms, add_loops=add_loops)
        self._carve_corridors(grid, rooms, graph)
        return grid, rooms, graph

    def generate(self, config, max_retries=30):
        md = None
        for _ in range(max_retries):
            md = self._generate_once(config)
            ok, _ = self.validate(md, config)
            if ok:
                return md
        return md

    def _generate_once(self, config):
        nf = config.get("n_floors", 1)
        w = config.get("width", 12)
        h = config.get("height", 12)
        has_doors = bool(config.get("locks"))

        # BSP: giữ nguyên, chỉ sinh cấu trúc phòng/hành lang
        all_f, all_r, all_g = [], [], []
        for _ in range(nf):
            grid, rooms, graph = self._gen_floor(
                w, h, add_loops=not has_doors
            )
            all_f.append(grid)
            all_r.append(rooms)
            all_g.append(graph)

        # Mỗi floor một set ô đã dùng, tránh trùng và dễ kiểm soát
        used_per_floor = [set() for _ in range(nf)]

        # ─── 1) Cầu thang: từng cặp tầng, một vị trí nối fi ↔ fi+1 ───
        for fi in range(nf - 1):
            bd, stx, sty = float("inf"), w // 2, h // 2
            for ra in all_r[fi]:
                ac = (ra[0] + ra[2] // 2, ra[1] + ra[3] // 2)
                for rb in all_r[fi + 1]:
                    bc = (rb[0] + rb[2] // 2, rb[1] + rb[3] // 2)
                    d = abs(ac[0] - bc[0]) + abs(ac[1] - bc[1])
                    if d < bd:
                        bd = d
                        stx = max(1, min(w - 2, ac[0]))
                        sty = max(1, min(h - 2, ac[1]))
            all_f[fi][sty][stx] = Tile.STAIR_UP
            all_f[fi + 1][sty][stx] = Tile.STAIR_DOWN
            used_per_floor[fi].add((stx, sty))
            used_per_floor[fi + 1].add((stx, sty))

        sf = config.get("start_floor", nf - 1)
        gf = config.get("goal_floor", 0)

        # ─── 2) Start: một ô trên start_floor ───
        sr = all_r[sf]
        sp = (
            self._pick_in_room(all_f[sf], sr[0], used_per_floor[sf])
            if sr
            else self._pick_floor(all_f[sf], used_per_floor[sf])
        )
        used_per_floor[sf].add(sp)

        # ─── 3) Goal: một ô trên goal_floor, xa start ───
        gr = all_r[gf]
        if gr:
            bd2, br = -1, gr[0]
            for r in gr:
                rc = (r[0] + r[2] // 2, r[1] + r[3] // 2)
                d = abs(rc[0] - sp[0]) + abs(rc[1] - sp[1])
                if sf != gf:
                    d += w
                if d > bd2:
                    bd2, br = d, r
            gp = self._pick_in_room(all_f[gf], br, used_per_floor[gf])
        else:
            gp = self._pick_floor(all_f[gf], used_per_floor[gf])
        used_per_floor[gf].add(gp)

        all_f[sf][sp[1]][sp[0]] = Tile.START
        all_f[gf][gp[1]][gp[0]] = Tile.GOAL
        start = (sp[0], sp[1], sf)
        goal = (gp[0], gp[1], gf)

        # ─── 4) Cửa + chìa: từng lock một ───
        for lock in config.get("locks", []):
            kf = lock.get("k", 0)
            df = lock.get("d", 0)
            ok = self._place_lock(
                all_f, all_r, all_g, kf, df, start, goal, used_per_floor
            )
            if not ok:
                used_df = used_per_floor[df]
                used_kf = used_per_floor[kf]
                kr = all_r[kf]
                dr = all_r[df]
                if kr:
                    kt = self._pick_in_room(
                        all_f[kf],
                        kr[self.rng.randint(0, len(kr) - 1)],
                        used_kf,
                    )
                else:
                    kt = self._pick_floor(all_f[kf], used_kf)
                all_f[kf][kt[1]][kt[0]] = Tile.KEY
                if dr:
                    dt = self._pick_in_room(
                        all_f[df],
                        dr[self.rng.randint(0, len(dr) - 1)],
                        used_df,
                    )
                else:
                    dt = self._pick_floor(all_f[df], used_df)
                all_f[df][dt[1]][dt[0]] = Tile.DOOR

        # ─── 5) Bẫy / máu / đạn: đặt từng ô một, có zone và không trùng ───
        for fi in range(nf):
            grid = all_f[fi]
            rooms = all_r[fi]
            graph = all_g[fi]
            used = used_per_floor[fi]

            if len(rooms) < 2:
                for _ in range(config.get("n_traps", 0)):
                    t = self._pick_floor(grid, used)
                    if grid[t[1]][t[0]] == Tile.FLOOR:
                        grid[t[1]][t[0]] = Tile.TRAP
                for _ in range(config.get("n_health", 0)):
                    t = self._pick_floor(grid, used)
                    if grid[t[1]][t[0]] == Tile.FLOOR:
                        grid[t[1]][t[0]] = Tile.HEALTH_PACK
                for _ in range(config.get("n_ammo", 0)):
                    t = self._pick_floor(grid, used)
                    if grid[t[1]][t[0]] == Tile.FLOOR:
                        grid[t[1]][t[0]] = Tile.AMMO_PACK
                continue

            si, gi = 0, len(rooms) - 1
            if fi == sf:
                for ri, r in enumerate(rooms):
                    if r[0] <= sp[0] < r[0] + r[2] and r[1] <= sp[1] < r[1] + r[3]:
                        si = ri
                        break
            if fi == gf:
                for ri, r in enumerate(rooms):
                    if r[0] <= gp[0] < r[0] + r[2] and r[1] <= gp[1] < r[1] + r[3]:
                        gi = ri
                        break
            ne = sum(1 for ec in config.get("enemies", []) if ec.get("floor", 0) == fi)
            cp = self._critical_path(graph, si, gi, len(rooms))
            zones = self._assign_zones(rooms, graph, cp, ne)

            n_traps = config.get("n_traps", 0)
            n_health = config.get("n_health", 0)
            n_ammo = config.get("n_ammo", 0)
            # Mỗi loại build danh sách ô hợp lệ 1 lần rồi shuffle + pop → nhanh hơn
            trap_cands = self._collect_candidates(
                grid, rooms, zones, ("passage", "combat", "treasure"), used
            )
            self.rng.shuffle(trap_cands)
            for _ in range(n_traps):
                if trap_cands:
                    c = trap_cands.pop()
                    grid[c[1]][c[0]] = Tile.TRAP
                    used.add(c)
            health_cands = self._collect_candidates(
                grid, rooms, zones, ("treasure", "passage", "spawn"), used
            )
            self.rng.shuffle(health_cands)
            for _ in range(n_health):
                if health_cands:
                    c = health_cands.pop()
                    grid[c[1]][c[0]] = Tile.HEALTH_PACK
                    used.add(c)
            ammo_cands = self._collect_candidates(
                grid, rooms, zones, ("treasure", "combat", "passage"), used
            )
            self.rng.shuffle(ammo_cands)
            for _ in range(n_ammo):
                if ammo_cands:
                    c = ammo_cands.pop()
                    grid[c[1]][c[0]] = Tile.AMMO_PACK
                    used.add(c)

        return {
            "floors": all_f,
            "n_floors": nf,
            "width": w,
            "height": h,
            "start": start,
            "goal": goal,
        }


# ═══════════════════════════════════════════════════════════════
# Entity System
# ═══════════════════════════════════════════════════════════════


class Projectile:
    __slots__ = ("x", "y", "floor", "dx", "dy", "owner", "ric", "alive")

    def __init__(self, x, y, floor, dx, dy, owner="agent", ric=0):
        self.x, self.y, self.floor = x, y, floor
        self.dx, self.dy = dx, dy
        self.owner = owner
        self.ric = ric
        self.alive = True

    def move(self):
        self.x += self.dx
        self.y += self.dy


class PatrolEnemy:
    def __init__(self, x, y, floor, waypoints=None, vision=4):
        self.x, self.y, self.floor = x, y, floor
        self.etype = "patrol"
        self.hp = 1
        self.alive = True
        self.wps = waypoints or [(x, y)]
        self.wp_idx = 0
        self.vision = vision
        self.state = "patrol"
        self.inv_target = None
        self.inv_timer = 0
        self.old_x, self.old_y = x, y

    def update(self, grid, apos, afloor, noise, npos):
        if not self.alive:
            return
        self.old_x, self.old_y = self.x, self.y
        if self.floor != afloor:
            self._patrol(grid)
            return
        ax, ay = apos
        blocked = {(ax, ay)}
        see = self._los(grid, ax, ay) and self._d(ax, ay) <= self.vision
        hear = (
            noise > 3
            and npos
            and abs(npos[0] - self.x) + abs(npos[1] - self.y) <= noise
        )
        if see:
            self.state = "investigate"
            self.inv_target = (ax, ay)
            self.inv_timer = 10
        elif hear:
            self.state = "investigate"
            self.inv_target = npos
            self.inv_timer = 8
        if self.state == "patrol":
            self._patrol(grid, blocked)
        elif self.state == "investigate":
            if self.inv_target:
                self._toward(grid, *self.inv_target, blocked=blocked)
            self.inv_timer -= 1
            if self.inv_timer <= 0:
                self.state = "patrol"

    def _patrol(self, grid, blocked=None):
        if not self.wps:
            return
        tx, ty = self.wps[self.wp_idx]
        if self.x == tx and self.y == ty:
            self.wp_idx = (self.wp_idx + 1) % len(self.wps)
            tx, ty = self.wps[self.wp_idx]
        self._toward(grid, tx, ty, blocked=blocked)

    def _toward(self, grid, tx, ty, blocked=None):
        bd = abs(tx - self.x) + abs(ty - self.y)
        best = None
        h, w = grid.shape
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = self.x + dx, self.y + dy
            if (
                0 <= nx < w
                and 0 <= ny < h
                and grid[ny][nx] not in (Tile.WALL, Tile.DOOR)
            ):
                if blocked and (nx, ny) in blocked:
                    continue
                d = abs(tx - nx) + abs(ty - ny)
                if d < bd:
                    bd, best = d, (nx, ny)
        if best:
            self.x, self.y = best

    def _los(self, grid, tx, ty):
        x0, y0 = self.x, self.y
        ddx, ddy = abs(tx - x0), abs(ty - y0)
        sx = 1 if tx > x0 else -1
        sy = 1 if ty > y0 else -1
        err = ddx - ddy
        h, w = grid.shape
        for _ in range(max(ddx, ddy) + 2):
            if x0 == tx and y0 == ty:
                return True
            if not (0 <= x0 < w and 0 <= y0 < h):
                return False
            if grid[y0][x0] in (Tile.WALL, Tile.DOOR) and (
                x0,
                y0,
            ) != (self.x, self.y):
                return False
            e2 = 2 * err
            if e2 > -ddy:
                err -= ddy
                x0 += sx
            if e2 < ddx:
                err += ddx
                y0 += sy
        return False

    def _d(self, ax, ay):
        return abs(ax - self.x) + abs(ay - self.y)


class ChaserEnemy:
    def __init__(self, x, y, floor, det=6):
        self.x, self.y, self.floor = x, y, floor
        self.etype = "chaser"
        self.hp = 2
        self.alive = True
        self.det = det
        self.state = "idle"
        self.timer = 0
        self.old_x, self.old_y = x, y

    def update(self, grid, apos, afloor, noise, npos):
        if not self.alive or self.floor != afloor:
            return
        self.old_x, self.old_y = self.x, self.y
        ax, ay = apos
        dist = abs(ax - self.x) + abs(ay - self.y)
        if dist <= self.det:
            self.state = "chase"
            self.timer = 15
        elif (
            noise > 5
            and npos
            and abs(npos[0] - self.x) + abs(npos[1] - self.y) <= noise
        ):
            self.state = "chase"
            self.timer = 10
        if self.state == "chase":
            path = self._astar(grid, (self.x, self.y), (ax, ay))
            if path and len(path) > 1:
                nx, ny = path[1]
                if (nx, ny) != (ax, ay):
                    self.x, self.y = nx, ny
            self.timer -= 1
            if self.timer <= 0:
                self.state = "idle"

    def _astar(self, grid, start, goal):
        h, w = grid.shape
        op = [(0, start)]
        came = {}
        gs = {start: 0}
        while op:
            _, cur = heapq.heappop(op)
            if cur == goal:
                p = [cur]
                while cur in came:
                    cur = came[cur]
                    p.append(cur)
                return p[::-1]
            for ddx, ddy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = cur[0] + ddx, cur[1] + ddy
                if (
                    0 <= nx < w
                    and 0 <= ny < h
                    and grid[ny][nx] not in (Tile.WALL, Tile.DOOR)
                ):
                    tg = gs[cur] + 1
                    if (nx, ny) not in gs or tg < gs[(nx, ny)]:
                        gs[(nx, ny)] = tg
                        heapq.heappush(
                            op,
                            (
                                tg
                                + abs(nx - goal[0])
                                + abs(ny - goal[1]),
                                (nx, ny),
                            ),
                        )
                        came[(nx, ny)] = cur
        return None


class SniperEnemy:
    def __init__(self, x, y, floor):
        self.x, self.y, self.floor = x, y, floor
        self.etype = "sniper"
        self.hp = 1
        self.alive = True
        self.dir = (1, 0)
        self.charge = 0
        self.cd = 0
        self.laser_path = []
        self.charging = False
        self.old_x, self.old_y = x, y

    def update(self, grid, apos, afloor, noise, npos):
        self.laser_path = []
        proj = None
        if not self.alive or self.floor != afloor:
            return proj
        self.old_x, self.old_y = self.x, self.y
        if self.cd > 0:
            self.cd -= 1
            self.charging = False
            return proj
        ax, ay = apos
        dx, dy = ax - self.x, ay - self.y
        if abs(dx) >= abs(dy) and dx != 0:
            self.dir = (1 if dx > 0 else -1, 0)
        elif dy != 0:
            self.dir = (0, 1 if dy > 0 else -1)
        in_line = (self.dir[0] != 0 and dy == 0 and dx * self.dir[0] > 0) or (
            self.dir[1] != 0 and dx == 0 and dy * self.dir[1] > 0
        )
        if in_line:
            self.charging = True
            self.charge += 1
            h, w = grid.shape
            lx, ly = self.x, self.y
            while True:
                lx += self.dir[0]
                ly += self.dir[1]
                if (
                    not (0 <= lx < w and 0 <= ly < h)
                    or grid[ly][lx] in (Tile.WALL, Tile.DOOR)
                ):
                    break
                self.laser_path.append((lx, ly))
            if self.charge >= 3:
                proj = Projectile(
                    self.x,
                    self.y,
                    self.floor,
                    self.dir[0],
                    self.dir[1],
                    "sniper",
                    0,
                )
                self.charge = 0
                self.cd = 3
                self.charging = False
        else:
            self.charge = max(0, self.charge - 1)
            self.charging = False
        return proj


class AgentState:
    def __init__(self, x, y, floor, hp=3, ammo=3, stamina=3):
        self.x, self.y, self.floor = x, y, floor
        self.hp, self.max_hp = hp, hp
        self.ammo, self.max_ammo = ammo, max(5, ammo)
        self.stamina, self.max_stamina = stamina, stamina
        self.noise_level = 0
        self.keys = 0
        self.last_dir = (1, 0)
        self.alive = True
        self.visited = set()
        self.steps = 0
        self.old_x, self.old_y = x, y

    def damage(self, n=1):
        self.hp -= n
        self.alive = self.hp > 0

    def heal(self, n=1):
        self.hp = min(self.max_hp, self.hp + n)

    def add_ammo(self, n=2):
        self.ammo = min(self.max_ammo, self.ammo + n)

    def use_ammo(self):
        if self.ammo > 0:
            self.ammo -= 1
            return True
        return False

    def use_stamina(self):
        if self.stamina > 0:
            self.stamina -= 1
            return True
        return False

    def regen_stamina(self):
        self.stamina = min(self.max_stamina, self.stamina + 1)


# ═══════════════════════════════════════════════════════════════
# Physics & Mechanics
# ═══════════════════════════════════════════════════════════════


class BallisticsEngine:
    def __init__(self):
        self.bullets = []

    def fire(self, x, y, f, dx, dy, owner="agent", ric=1):
        self.bullets.append(Projectile(x, y, f, dx, dy, owner, ric))

    def add(self, p):
        if p:
            self.bullets.append(p)

    def update(self, grid, enemies, agent):
        events = []
        alive = []
        h, w = grid.shape
        for b in self.bullets:
            if not b.alive:
                continue
            b.move()
            if not (0 <= b.x < w and 0 <= b.y < h):
                b.alive = False
                continue
            if grid[b.y][b.x] in (Tile.WALL, Tile.DOOR):
                if b.ric > 0:
                    b.ric -= 1
                    if (
                        0 <= b.x - b.dx < w
                        and grid[b.y][b.x - b.dx]
                        not in (Tile.WALL, Tile.DOOR)
                    ):
                        b.dx = -b.dx
                    else:
                        b.dy = -b.dy
                    b.x += b.dx
                    b.y += b.dy
                    if not (0 <= b.x < w and 0 <= b.y < h) or grid[b.y][
                        b.x
                    ] in (Tile.WALL, Tile.DOOR):
                        b.alive = False
                else:
                    b.alive = False
                if b.alive:
                    alive.append(b)
                continue
            if b.owner == "agent":
                for e in enemies:
                    if (
                        e.alive
                        and e.x == b.x
                        and e.y == b.y
                        and e.floor == b.floor
                    ):
                        e.hp -= 1
                        if e.hp <= 0:
                            e.alive = False
                            events.append(("kill", e))
                        b.alive = False
                        break
            elif (
                b.x == agent.x
                and b.y == agent.y
                and b.floor == agent.floor
            ):
                events.append(("agent_hit", b))
                b.alive = False
            if b.alive:
                alive.append(b)
        self.bullets = alive
        return events

    def positions(self, f):
        return [
            (b.x, b.y, b.owner)
            for b in self.bullets
            if b.alive and b.floor == f
        ]


class StealthSystem:
    def __init__(self):
        self.noise = 0
        self.npos = None

    def calc(self, action, agent):
        if action == 8:
            n = 5
        elif 4 <= action <= 7:
            n = 10
        else:
            n = 0
        self.noise = n
        self.npos = (agent.x, agent.y) if n > 0 else None
        agent.noise_level = n
        return n

    def decay(self, agent):
        agent.noise_level = max(0, agent.noise_level - 2)


# ═══════════════════════════════════════════════════════════════
# Optimized Gym Environment
# Key change: approach reward uses cached BFS, not per-step BFS
# ═══════════════════════════════════════════════════════════════

N_CHANNELS = 16
N_VEC = 15


def _build_obs(grid, agent, enemies, bpos, lasers, nf, target_3d, max_steps, vr=4):
    """Build (16, V, V) visual + (15,) vector observation."""
    vs = 2 * vr + 1
    h, w = grid.shape
    obs = np.zeros((N_CHANNELS, vs, vs), dtype=np.float32)

    for dy in range(-vr, vr + 1):
        for dx in range(-vr, vr + 1):
            gx, gy = agent.x + dx, agent.y + dy
            oy, ox = dy + vr, dx + vr
            if 0 <= gx < w and 0 <= gy < h:
                t = int(grid[gy][gx])
                ch_map = {
                    int(Tile.WALL): 0,
                    int(Tile.TRAP): 1,
                    int(Tile.DOOR): 2,
                    int(Tile.STAIR_UP): 3,
                    int(Tile.STAIR_DOWN): 4,
                    int(Tile.GOAL): 5,
                    int(Tile.KEY): 6,
                    int(Tile.HEALTH_PACK): 7,
                    int(Tile.AMMO_PACK): 8,
                }
                if t in ch_map:
                    obs[ch_map[t]][oy][ox] = 1.0
                if (gx, gy, agent.floor) in agent.visited:
                    obs[15][oy][ox] = 1.0
            else:
                obs[0][oy][ox] = 1.0

    obs[9][vr][vr] = 1.0

    ETYPE_CH = {"patrol": 10, "chaser": 11, "sniper": 12}
    min_enemy_dist = float("inf")
    for e in enemies:
        if e.alive and e.floor == agent.floor:
            ed = abs(e.x - agent.x) + abs(e.y - agent.y)
            if ed < min_enemy_dist:
                min_enemy_dist = ed
            ox, oy = e.x - agent.x + vr, e.y - agent.y + vr
            if 0 <= ox < vs and 0 <= oy < vs:
                ch = ETYPE_CH.get(e.etype, 10)
                obs[ch][oy][ox] = 1.0

    for bx, by, owner in bpos:
        ox, oy = bx - agent.x + vr, by - agent.y + vr
        if 0 <= ox < vs and 0 <= oy < vs:
            obs[13][oy][ox] = 1.0
    for lx, ly in lasers:
        ox, oy = lx - agent.x + vr, ly - agent.y + vr
        if 0 <= ox < vs and 0 <= oy < vs:
            obs[14][oy][ox] = 1.0

    gx, gy, gf = target_3d
    if gf == agent.floor:
        ref_x, ref_y = gx, gy
    else:
        target_stair = Tile.STAIR_UP if gf > agent.floor else Tile.STAIR_DOWN
        best_d, ref_x, ref_y = float("inf"), agent.x, agent.y
        for sy in range(h):
            for sx in range(w):
                if int(grid[sy][sx]) == int(target_stair):
                    d = abs(sx - agent.x) + abs(sy - agent.y)
                    if d < best_d:
                        best_d, ref_x, ref_y = d, sx, sy

    local_visited_density = float(obs[15].mean() * 2.0 - 1.0)
    enemy_prox = (
        -1.0
        if min_enemy_dist == float("inf")
        else (1.0 - min_enemy_dist / max(1, h + w)) * 2.0 - 1.0
    )
    time_rem = (1.0 - agent.steps / max(1, max_steps)) * 2.0 - 1.0
    cur_tile = int(grid[agent.y][agent.x])
    on_stair = (
        1.0
        if cur_tile in (Tile.STAIR_UP, Tile.STAIR_DOWN)
        else -1.0
    )
    floor_ratio = (
        (agent.floor / max(1, nf - 1) if nf > 1 else 0.0) * 2.0 - 1.0
    )
    key_ratio = (
        (agent.keys / max(1, agent.keys, 1)) * 2.0 - 1.0
    )

    vec = np.array(
        [
            (agent.hp / max(1, agent.max_hp)) * 2.0 - 1.0,
            (agent.ammo / max(1, agent.max_ammo)) * 2.0 - 1.0,
            (agent.stamina / max(1, agent.max_stamina)) * 2.0 - 1.0,
            np.clip(key_ratio, -1.0, 1.0),
            np.clip((agent.noise_level / 10.0) * 2.0 - 1.0, -1.0, 1.0),
            np.clip(floor_ratio, -1.0, 1.0),
            float(np.sign(ref_x - agent.x)),
            float(np.sign(ref_y - agent.y)),
            float(np.sign(gf - agent.floor)),
            float(agent.last_dir[0]),
            float(agent.last_dir[1]),
            np.clip(time_rem, -1.0, 1.0),
            np.clip(local_visited_density, -1.0, 1.0),
            on_stair,
            np.clip(enemy_prox, -1.0, 1.0),
        ],
        dtype=np.float32,
    )
    return obs, vec


class MazeEnv(gym.Env):
    """Optimized Multi-Floor Maze for local training.

    Key optimization vs original: approach reward uses cached distance
    that only recomputes when key/door state changes, not every step.
    """

    metadata = {"render_modes": ["rgb_array"]}

    REWARD = {
        "step": -0.02,
        "wall": -0.12,
        "goal": 25.0,
        "key": 2.5,
        "door": 3.5,
        "door_lk": -0.15,
        "kill": 1.25,
        "dmg": -2.0,
        "death": -18.0,
        "stairs": 0.75,
        "explore": 0.05,
        "trap": -2.5,
        "hp": 0.75,
        "ammo": 0.50,
        "approach": 0.10,
        "shoot": -0.04,
    }

    def __init__(self, config=None, render_mode="rgb_array"):
        super().__init__()
        self.cfg = config or {}
        self.render_mode = render_mode
        self.vr = self.cfg.get("view_radius", 4)
        vs = 2 * self.vr + 1
        self.observation_space = spaces.Dict(
            {
                "visual": spaces.Box(0, 1, (N_CHANNELS, vs, vs), np.float32),
                "vector": spaces.Box(-1, 1, (N_VEC,), np.float32),
            }
        )
        self.action_space = spaces.Discrete(10)
        self.R = copy.deepcopy(self.REWARD)
        self.R.update(self.cfg.get("rewards", {}))
        self.max_steps = self.cfg.get("max_steps", 200)
        self.floors = None
        self.agent = None
        self.enemies = []
        self.goal = (0, 0, 0)
        self.done = False
        self.win = False
        self.total_reward = 0
        self._lasers = []
        self._cached_dist = 999
        self._cache_key_state = None
        self.max_theoretical_reward = 0.0

    def _estimate_reward_upper_bound(self, map_data):
        """Ước lượng upper bound reward có thể lấy được trên map này.

        Bỏ qua penalty bước đi / trap / đạn, chỉ cộng:
        - reward goal
        - reward nhặt key / máu / đạn
        - reward stairs (mỗi ô cầu thang)
        - reward explore cho mọi ô có thể ghé qua (bfs không chặn cửa)
        """
        floors = map_data["floors"]
        sx, sy, sf = map_data["start"]
        goal = map_data["goal"]
        pmg = ProceduralMapGenerator()
        reachable = pmg._bfs_reachable_set(floors, (sx, sy, sf), set())
        tiles = set(reachable)
        R = self.R
        val = 0.0
        for (x, y, f) in tiles:
            t = int(floors[f][y][x])
            if t == int(Tile.KEY):
                val += R.get("key", 0.0)
            elif t == int(Tile.HEALTH_PACK):
                val += R.get("hp", 0.0)
            elif t == int(Tile.AMMO_PACK):
                val += R.get("ammo", 0.0)
            elif t in (int(Tile.STAIR_UP), int(Tile.STAIR_DOWN)):
                val += R.get("stairs", 0.0)
            elif t == int(Tile.GOAL):
                val += R.get("goal", 0.0)
        val += R.get("explore", 0.0) * len(tiles)
        # đảm bảo goal nếu reachable thì đã tính
        if goal not in tiles:
            gx, gy, gf = goal
            if 0 <= gf < len(floors):
                if 0 <= gy < floors[gf].shape[0] and 0 <= gx < floors[gf].shape[1]:
                    if int(floors[gf][gy][gx]) == int(Tile.GOAL):
                        val += R.get("goal", 0.0)
        return float(val)

    def _compute_distance(self):
        """BFS distance from agent to dynamic goal. Called sparingly."""
        dg = self._dynamic_goal()
        gx, gy, gf = dg
        ax, ay, af = self.agent.x, self.agent.y, self.agent.floor
        if ax == gx and ay == gy and af == gf:
            return 0
        h, w = self.floors[0].shape
        vis = {(ax, ay, af)}
        q = deque([(ax, ay, af, 0)])
        while q:
            cx, cy, cf, d = q.popleft()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < w and 0 <= ny < h and (nx, ny, cf) not in vis:
                    t = int(self.floors[cf][ny][nx])
                    if t != int(Tile.WALL) and not (
                        t == int(Tile.DOOR) and self.agent.keys == 0
                    ):
                        if nx == gx and ny == gy and cf == gf:
                            return d + 1
                        vis.add((nx, ny, cf))
                        q.append((nx, ny, cf, d + 1))
            ct = int(self.floors[cf][cy][cx])
            if ct == int(Tile.STAIR_UP) and cf < len(self.floors) - 1:
                if (cx, cy, cf + 1) not in vis:
                    if cx == gx and cy == gy and cf + 1 == gf:
                        return d + 1
                    vis.add((cx, cy, cf + 1))
                    q.append((cx, cy, cf + 1, d + 1))
            elif ct == int(Tile.STAIR_DOWN) and cf > 0:
                if (cx, cy, cf - 1) not in vis:
                    if cx == gx and cy == gy and cf - 1 == gf:
                        return d + 1
                    vis.add((cx, cy, cf - 1))
                    q.append((cx, cy, cf - 1, d + 1))
        return 999

    def _key_door_state(self):
        """Hashable representation of key/door state for cache invalidation."""
        return (self.agent.keys, self.agent.x, self.agent.y, self.agent.floor)

    def _update_distance_cache(self, force=False):
        new_state = self._key_door_state()
        if force or new_state != self._cache_key_state:
            self._cached_dist = self._compute_distance()
            self._cache_key_state = new_state

    def _dynamic_goal(self):
        if self.agent.keys == 0:
            bd, bp = float("inf"), None
            for f, grid in enumerate(self.floors):
                h, w = grid.shape
                for sy in range(h):
                    for sx in range(w):
                        if grid[sy][sx] == Tile.KEY:
                            d = (
                                abs(sx - self.agent.x)
                                + abs(sy - self.agent.y)
                                + abs(f - self.agent.floor) * 20
                            )
                            if d < bd:
                                bd, bp = d, (sx, sy, f)
            if bp:
                return bp
        return self.goal

    def reset(self, seed=None, options=None):
        if options and "map_data" in options:
            md = options["map_data"]
        else:
            mg = ProceduralMapGenerator(seed)
            md = mg.generate(self.cfg)
        # estimate trước khi copy floors (dùng cho phân tích / baseline)
        self.max_theoretical_reward = self._estimate_reward_upper_bound(md)
        self.floors = [g.copy() for g in md["floors"]]
        self.nf = md["n_floors"]
        self.mw, self.mh = md["width"], md["height"]
        sx, sy, sf = md["start"]
        self.goal = md["goal"]

        hp = self.cfg.get("agent_hp", 3)
        ammo = self.cfg.get("agent_ammo", 3)
        stam = self.cfg.get("agent_stamina", 3)
        self.agent = AgentState(sx, sy, sf, hp, ammo, stam)
        self.agent.max_ammo = self.cfg.get("agent_max_ammo", 5)

        self.enemies = []
        for ec in self.cfg.get("enemies", []):
            ef = ec.get("floor", 0)
            g = self.floors[ef]
            ft = [
                (x, y)
                for y in range(self.mh)
                for x in range(self.mw)
                if g[y][x] == Tile.FLOOR and (x, y) != (sx, sy)
            ]
            ex, ey = random.choice(ft) if ft else (1, 1)
            et = ec.get("type", "patrol")
            if et == "patrol":
                wps = [(ex, ey)]
                for _ in range(3):
                    wps.append(
                        (
                            max(1, min(self.mw - 2, ex + random.randint(-4, 4))),
                            max(1, min(self.mh - 2, ey + random.randint(-4, 4))),
                        )
                    )
                self.enemies.append(PatrolEnemy(ex, ey, ef, wps))
            elif et == "chaser":
                self.enemies.append(ChaserEnemy(ex, ey, ef))
            elif et == "sniper":
                self.enemies.append(SniperEnemy(ex, ey, ef))

        self.ballistics = BallisticsEngine()
        self.stealth = StealthSystem()
        self.done = False
        self.win = False
        self.total_reward = 0
        self.agent.steps = 0
        self.agent.visited = {(sx, sy, sf)}
        self._lasers = []
        self._cache_key_state = None
        self._update_distance_cache(force=True)

        return self._obs(), self._info()

    def step(self, action):
        if self.done:
            return self._obs(), 0.0, True, False, self._info()

        dist_old = self._cached_dist
        self.agent.old_x, self.agent.old_y = self.agent.x, self.agent.y

        a = int(action)
        self.agent.steps += 1
        r = self.R["step"]
        g = self.floors[self.agent.floor]
        self.stealth.calc(a, self.agent)
        key_door_changed = False

        if a in (0, 1, 2, 3):
            dx, dy = DIR_DELTA[a]
            self.agent.last_dir = (dx, dy)
            mr, kdc = self._try_move(self.agent.x + dx, self.agent.y + dy)
            r += mr
            key_door_changed = kdc
        elif 4 <= a <= 7:
            dx, dy = DIR_DELTA[a - 4]
            self.agent.last_dir = (dx, dy)
            if self.agent.use_ammo():
                r += self.R["shoot"]
                self.ballistics.fire(
                    self.agent.x,
                    self.agent.y,
                    self.agent.floor,
                    dx,
                    dy,
                    "agent",
                    self.cfg.get("ricochet", 0),
                )
            else:
                r += self.R["wall"]
        elif a == 8:
            if self.agent.use_stamina():
                dx, dy = self.agent.last_dir
                for _ in range(2):
                    nx, ny = self.agent.x + dx, self.agent.y + dy
                    mr, kdc = self._try_move(nx, ny)
                    r += mr
                    key_door_changed = key_door_changed or kdc
                    # nếu bị chặn (tường/cửa không mở) thì dừng dash
                    if (self.agent.x, self.agent.y) != (nx, ny):
                        break
            else:
                r += self.R["wall"]
        elif a == 9:
            t = g[self.agent.y][self.agent.x]
            if t == Tile.STAIR_UP and self.agent.floor < self.nf - 1:
                self.agent.floor += 1
                pk2 = (self.agent.x, self.agent.y, self.agent.floor)
                if pk2 not in self.agent.visited:
                    r += self.R["stairs"]
                key_door_changed = True
            elif t == Tile.STAIR_DOWN and self.agent.floor > 0:
                self.agent.floor -= 1
                pk2 = (self.agent.x, self.agent.y, self.agent.floor)
                if pk2 not in self.agent.visited:
                    r += self.R["stairs"]
                key_door_changed = True
            else:
                r += self.R["wall"]

        pk = (self.agent.x, self.agent.y, self.agent.floor)
        if pk not in self.agent.visited:
            self.agent.visited.add(pk)
            r += self.R["explore"]

        # Approach reward: use cached distance, only recompute when needed
        self._update_distance_cache(force=key_door_changed)
        dist_new = self._cached_dist
        if self.R.get("approach", 0) > 0:
            r += self.R["approach"] * np.clip(dist_old - dist_new, -1, 1)

        for ev, _ in self.ballistics.update(g, self.enemies, self.agent):
            if ev == "kill":
                r += self.R["kill"]
            elif ev == "agent_hit":
                self.agent.damage(1)
                r += self.R["dmg"]

        self._lasers = []
        npos = self.stealth.npos
        for e in self.enemies:
            eg = self.floors[e.floor] if e.floor < self.nf else g
            if isinstance(e, SniperEnemy):
                proj = e.update(
                    eg,
                    (self.agent.x, self.agent.y),
                    self.agent.floor,
                    self.agent.noise_level,
                    npos,
                )
                if proj:
                    self.ballistics.add(proj)
                self._lasers.extend(e.laser_path)
            else:
                e.update(
                    eg,
                    (self.agent.x, self.agent.y),
                    self.agent.floor,
                    self.agent.noise_level,
                    npos,
                )

        for e in self.enemies:
            if (
                e.alive
                and e.floor == self.agent.floor
                and abs(e.x - self.agent.x) + abs(e.y - self.agent.y) <= 1
                and hasattr(e, "state")
                and e.state in ("investigate", "chase")
            ):
                self.agent.damage(1)
                r += self.R["dmg"]

        if self.agent.steps % 5 == 0:
            self.agent.regen_stamina()
        self.stealth.decay(self.agent)

        term = False
        gx, gy, gf = self.goal
        if not self.agent.alive:
            r += self.R["death"]
            term = True
        if (
            self.agent.x == gx
            and self.agent.y == gy
            and self.agent.floor == gf
        ):
            r += self.R["goal"]
            term = True
            self.win = True

        trunc = self.agent.steps >= self.max_steps
        self.done = term or trunc
        self.total_reward += r
        return self._obs(), float(r), term, trunc, self._info()

    def _try_move(self, nx, ny):
        """Returns (reward, key_door_changed)."""
        g = self.floors[self.agent.floor]
        h, w = g.shape
        if not (0 <= nx < w and 0 <= ny < h):
            return self.R["wall"], False
        t = int(g[ny][nx])
        if t == Tile.WALL:
            return self.R["wall"], False
        if t == Tile.DOOR:
            if self.agent.keys > 0:
                self.agent.keys -= 1
                g[ny][nx] = Tile.FLOOR
                self.agent.x, self.agent.y = nx, ny
                return self.R["door"], True
            return self.R["door_lk"], False
        self.agent.x, self.agent.y = nx, ny
        rv = 0.0
        kdc = False
        if t == Tile.TRAP:
            self.agent.damage(1)
            rv += self.R["trap"]
        elif t == Tile.KEY:
            self.agent.keys += 1
            g[ny][nx] = Tile.FLOOR
            rv += self.R["key"]
            kdc = True
        elif t == Tile.HEALTH_PACK:
            self.agent.heal(1)
            g[ny][nx] = Tile.FLOOR
            rv += self.R["hp"]
        elif t == Tile.AMMO_PACK:
            self.agent.add_ammo(2)
            g[ny][nx] = Tile.FLOOR
            rv += self.R["ammo"]
        return rv, kdc

    def _walk(self, nx, ny):
        g = self.floors[self.agent.floor]
        h, w = g.shape
        if not (0 <= nx < w and 0 <= ny < h):
            return False
        return int(g[ny][nx]) not in (Tile.WALL, Tile.DOOR)

    def _obs(self):
        g = self.floors[self.agent.floor]
        bp = self.ballistics.positions(self.agent.floor)
        dg = self._dynamic_goal()
        v, vec = _build_obs(
            g, self.agent, self.enemies, bp, self._lasers,
            self.nf, dg, self.max_steps, self.vr,
        )
        return {"visual": v, "vector": vec}

    def _info(self):
        return {
            "win": self.win,
            "steps": self.agent.steps,
            "hp": self.agent.hp,
            "ammo": self.agent.ammo,
            "floor": self.agent.floor,
            "keys": self.agent.keys,
            "total_reward": self.total_reward,
            "enemies_alive": sum(1 for e in self.enemies if e.alive),
            "explored": len(self.agent.visited),
            "max_theoretical_reward": getattr(
                self, "max_theoretical_reward", None
            ),
        }


# ═══════════════════════════════════════════════════════════════
# Curriculum
# ═══════════════════════════════════════════════════════════════

CURRICULUM = {
    1: {
        "name": "Crawl",
        "width": 7, "height": 7, "n_floors": 1, "max_steps": 50,
        "agent_hp": 10, "agent_ammo": 0, "agent_max_ammo": 0, "agent_stamina": 3,
        "n_traps": 0, "n_health": 2, "n_ammo": 0,
        "enemies": [], "locks": [],
        "start_floor": 0, "goal_floor": 0,
    },
    2: {
        "name": "Walk",
        "width": 9, "height": 9, "n_floors": 1, "max_steps": 60,
        "agent_hp": 10, "agent_ammo": 0, "agent_max_ammo": 0, "agent_stamina": 3,
        "n_traps": 0, "n_health": 2, "n_ammo": 0,
        "enemies": [], "locks": [],
        "start_floor": 0, "goal_floor": 0,
    },
    3: {
        "name": "Trapper",
        "width": 9, "height": 9, "n_floors": 1, "max_steps": 70,
        "agent_hp": 15, "agent_ammo": 0, "agent_max_ammo": 0, "agent_stamina": 3,
        "n_traps": 2, "n_health": 3, "n_ammo": 0,
        "enemies": [], "locks": [],
        "start_floor": 0, "goal_floor": 0,
    },
    4: {
        "name": "Locksmith",
        "width": 11, "height": 11, "n_floors": 1, "max_steps": 100,
        "agent_hp": 15, "agent_ammo": 0, "agent_max_ammo": 0, "agent_stamina": 3,
        "n_traps": 1, "n_health": 3, "n_ammo": 0,
        "enemies": [], "locks": [{"k": 0, "d": 0}],
        "start_floor": 0, "goal_floor": 0,
    },
    5: {
        "name": "Climber",
        "width": 11, "height": 11, "n_floors": 2, "max_steps": 150,
        "agent_hp": 20, "agent_ammo": 0, "agent_max_ammo": 0, "agent_stamina": 3,
        "n_traps": 2, "n_health": 4, "n_ammo": 0,
        "enemies": [], "locks": [],
        "start_floor": 1, "goal_floor": 0,
    },
    6: {
        "name": "Scout",
        "width": 11, "height": 11, "n_floors": 2, "max_steps": 180,
        "agent_hp": 20, "agent_ammo": 5, "agent_max_ammo": 5, "agent_stamina": 3,
        "n_traps": 2, "n_health": 5, "n_ammo": 1,
        "enemies": [{"type": "patrol", "floor": 0}],
        "locks": [],
        "start_floor": 1, "goal_floor": 0,
    },
    7: {
        "name": "Operative",
        "width": 11, "height": 11, "n_floors": 2, "max_steps": 250,
        "agent_hp": 10, "agent_ammo": 5, "agent_max_ammo": 5, "agent_stamina": 5,
        "n_traps": 3, "n_health": 3, "n_ammo": 2,
        "enemies": [
            {"type": "chaser", "floor": 0},
            {"type": "patrol", "floor": 1},
        ],
        "locks": [{"k": 0, "d": 0}],
        "start_floor": 1, "goal_floor": 0,
    },
    8: {
        "name": "Elite",
        "width": 13, "height": 13, "n_floors": 2, "max_steps": 350,
        "agent_hp": 10, "agent_ammo": 6, "agent_max_ammo": 6, "agent_stamina": 5,
        "n_traps": 3, "n_health": 3, "n_ammo": 3,
        "enemies": [
            {"type": "sniper", "floor": 0},
            {"type": "chaser", "floor": 0},
            {"type": "patrol", "floor": 1},
        ],
        "locks": [{"k": 0, "d": 0}, {"k": 1, "d": 1}],
        "start_floor": 1, "goal_floor": 0,
    },
    9: {
        "name": "Master",
        "width": 13, "height": 13, "n_floors": 3, "max_steps": 500,
        "agent_hp": 10, "agent_ammo": 8, "agent_max_ammo": 8, "agent_stamina": 6,
        "n_traps": 3, "n_health": 4, "n_ammo": 4,
        "enemies": [
            {"type": "sniper", "floor": 0},
            {"type": "chaser", "floor": 1},
            {"type": "patrol", "floor": 2},
            {"type": "chaser", "floor": 2},
        ],
        "locks": [{"k": 0, "d": 0}, {"k": 1, "d": 1}, {"k": 2, "d": 2}],
        "start_floor": 2, "goal_floor": 0,
    },
}
