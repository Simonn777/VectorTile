/** Small 2D explanation engine. This is not the measured Python/C++ engine. */
export function makePoints(count = 512, seed = 42) {
  let s = seed >>> 0;
  const random = () => { s = (1664525 * s + 1013904223) >>> 0; return s / 4294967296; };
  const centers = [[.28,.32],[.7,.34],[.58,.74]];
  return Array.from({length:count}, (_, id) => {
    const [cx,cy] = centers[id % centers.length];
    const normal = () => Math.sqrt(-2*Math.log(Math.max(random(),1e-12)))*Math.cos(2*Math.PI*random());
    return {id, x:Math.max(.03,Math.min(.97,cx+normal()*.13)), y:Math.max(.03,Math.min(.97,cy+normal()*.11))};
  });
}
export const compare = (a,b) => a.distance-b.distance || a.id-b.id;
export function mergeTile(best, points, query, k) {
  if (!Number.isInteger(k) || k < 1) throw new RangeError('k must be positive');
  return [...best,...points.map(p => ({id:p.id,distance:(p.x-query.x)**2+(p.y-query.y)**2}))].sort(compare).slice(0,k);
}
export function fullSearch(points, query, k) {return mergeTile([],points,query,k);}
