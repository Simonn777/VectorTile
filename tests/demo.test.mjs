import assert from 'node:assert/strict';
import test from 'node:test';
import {makePoints,mergeTile,fullSearch} from '../site/search.mjs';

test('seeded demonstration returns repeatable bounded vectors',()=>{
  assert.deepEqual(makePoints(),makePoints());
  for(const p of makePoints()){assert.ok(p.x>=0&&p.x<=1&&p.y>=0&&p.y<=1);}
});
test('tile size does not change the exhaustive answer',()=>{
  const points=makePoints(),q={x:.55,y:.48};
  const expected=points.map(p=>({id:p.id,distance:(p.x-q.x)**2+(p.y-q.y)**2})).sort((a,b)=>a.distance-b.distance||a.id-b.id).slice(0,10);
  for(const size of [1,7,16,64,128,513]){let best=[];for(let i=0;i<points.length;i+=size){best=mergeTile(best,points.slice(i,i+size),q,10);assert.ok(best.length<=10);}assert.deepEqual(best,expected);}
});
test('ties select lower vector ID even across tiles',()=>{
  const q={x:0,y:0},p=[{id:4,x:1,y:0},{id:2,x:0,y:1},{id:0,x:-1,y:0}];
  let best=mergeTile([],p.slice(0,2),q,2);best=mergeTile(best,p.slice(2),q,2);
  assert.deepEqual(best.map(x=>x.id),[0,2]);assert.deepEqual(best,fullSearch(p,q,2));
});
