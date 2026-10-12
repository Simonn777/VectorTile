/** Small 2D explanation engine. This is not the measured Python/C++ engine. */
export function makePoints(count = 512, seed = 42) {
  let s = seed >>> 0;
  const random = () => {
    s = (1664525 * s + 1013904223) >>> 0;
    return s / 4294967296;
  };
  const centers = [
    [0.28, 0.32],
    [0.7, 0.34],
    [0.58, 0.74],
  ];
  return Array.from({ length: count }, (_, id) => {
    const [cx, cy] = centers[id % centers.length];
    const normal = () =>
      Math.sqrt(-2 * Math.log(Math.max(random(), 1e-12))) *
      Math.cos(2 * Math.PI * random());
    return {
      id,
      x: Math.max(0.03, Math.min(0.97, cx + normal() * 0.13)),
      y: Math.max(0.03, Math.min(0.97, cy + normal() * 0.11)),
    };
  });
}
export const compare = (a, b) => a.distance - b.distance || a.id - b.id;
export function mergeTile(best, points, query, k) {
  if (!Number.isInteger(k) || k < 1) throw new RangeError("k must be positive");
  return [
    ...best,
    ...points.map((p) => ({
      id: p.id,
      distance: (p.x - query.x) ** 2 + (p.y - query.y) ** 2,
    })),
  ]
    .sort(compare)
    .slice(0, k);
}
export function fullSearch(points, query, k) {
  return mergeTile([], points, query, k);
}
