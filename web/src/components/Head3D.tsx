import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { Edge, Segment } from "../data";

interface Props {
  seg: Segment;
  edges: Edge[];
  litPath: { ch: number; via?: number | null }[];
  hot: number | null;
}

const css = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/** A scalp sphere with electrodes and directed edges drawn as arcs lifted off the surface. */
export default function Head3D({ seg, edges, litPath, hot }: Props) {
  const host = useRef<HTMLDivElement>(null);
  const api = useRef<{ setEdges: (e: Edge[], lit: Props["litPath"], hot: number | null) => void } | null>(null);

  useEffect(() => {
    const el = host.current!;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
    el.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 50);
    camera.position.set(2.7, -4.3, 3.1);
    camera.up.set(0, 0, 1);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.enablePan = false;
    controls.minDistance = 2.6;
    controls.maxDistance = 7;

    const ink = new THREE.Color(css("--ink"));
    const blue = new THREE.Color(css("--blue"));
    const red = new THREE.Color(css("--red"));
    const grid = new THREE.Color(css("--grid-major"));

    // scalp: a translucent paper shell ruled like the chart grid
    const shell = new THREE.Mesh(
      new THREE.SphereGeometry(0.985, 48, 32, 0, Math.PI * 2, 0, Math.PI * 0.62),
      new THREE.MeshBasicMaterial({ color: new THREE.Color(css("--paper")), transparent: true, opacity: 0.55, side: THREE.DoubleSide, depthWrite: false }),
    );
    shell.rotation.x = Math.PI / 2;
    scene.add(shell);
    const rule = new THREE.LineBasicMaterial({ color: grid, transparent: true, opacity: 0.9 });
    for (let k = 1; k <= 5; k++) {
      const th = (k / 6) * Math.PI * 0.62;
      const pts = Array.from({ length: 65 }, (_, i) => {
        const ph = (i / 64) * Math.PI * 2;
        return new THREE.Vector3(Math.sin(th) * Math.cos(ph), Math.sin(th) * Math.sin(ph), Math.cos(th)).multiplyScalar(0.99);
      });
      scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), rule));
    }
    for (let k = 0; k < 12; k++) {
      const ph = (k / 12) * Math.PI * 2;
      const pts = Array.from({ length: 33 }, (_, i) => {
        const th = (i / 32) * Math.PI * 0.62;
        return new THREE.Vector3(Math.sin(th) * Math.cos(ph), Math.sin(th) * Math.sin(ph), Math.cos(th)).multiplyScalar(0.99);
      });
      scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), rule));
    }
    // nose marker
    const nose = new THREE.Mesh(new THREE.ConeGeometry(0.07, 0.16, 16), new THREE.MeshBasicMaterial({ color: ink }));
    nose.position.set(0, 1.02, 0.18);
    nose.rotation.x = -Math.PI / 2;
    scene.add(nose);

    const pos = seg.pos3d.map(([x, y, z]) => new THREE.Vector3(x, y, Math.max(z, 0.05)).normalize());
    const nodes = pos.map((p) => {
      const m = new THREE.Mesh(new THREE.SphereGeometry(0.045, 16, 12), new THREE.MeshBasicMaterial({ color: ink }));
      m.position.copy(p);
      scene.add(m);
      return m;
    });

    const edgeGroup = new THREE.Group();
    scene.add(edgeGroup);
    const setEdges = (es: Edge[], lit: Props["litPath"], h: number | null) => {
      edgeGroup.children.forEach((c) => {
        const o = c as THREE.Mesh;
        o.geometry.dispose();
        (o.material as THREE.Material).dispose();
      });
      edgeGroup.clear();
      const wMax = Math.max(1e-6, ...es.map((e) => e.w));
      const chMax = Math.max(1e-6, ...es.map((e) => Math.abs(e.change)));
      const litSet = new Set(lit.map((l) => l.ch));
      nodes.forEach((n, i) => (n.material as THREE.MeshBasicMaterial).color.copy(litSet.has(i) ? red : i === h ? blue : ink));
      const draw = (s: number, d: number, color: THREE.Color, width: number, opacity: number) => {
        const a = pos[s];
        const b = pos[d];
        const mid = a.clone().add(b).multiplyScalar(0.5);
        const lift = 1 + 0.35 * a.distanceTo(b);
        const ctrl = mid.normalize().multiplyScalar(lift);
        const curve = new THREE.QuadraticBezierCurve3(a, ctrl, b);
        const tube = new THREE.Mesh(
          new THREE.TubeGeometry(curve, 24, width, 6, false),
          new THREE.MeshBasicMaterial({ color, transparent: true, opacity }),
        );
        edgeGroup.add(tube);
        const tip = curve.getPoint(0.93);
        const dir = curve.getTangent(0.93);
        const cone = new THREE.Mesh(new THREE.ConeGeometry(width * 3.2, width * 9, 10), new THREE.MeshBasicMaterial({ color, transparent: true, opacity }));
        cone.position.copy(tip);
        cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
        edgeGroup.add(cone);
      };
      for (const e of es) {
        const changed = e.change > 0.35 * chMax && e.change > 0.04;
        const faded = h != null && e.src !== h && e.dst !== h;
        draw(e.src, e.dst, changed ? red : blue, 0.004 + 0.012 * (e.w / wMax), faded ? 0.08 : 0.35 + 0.6 * (e.w / wMax));
      }
      for (const l of lit) if (l.via != null) draw(l.via, l.ch, red, 0.016, 1);
    };
    api.current = { setEdges };

    const resize = () => {
      const w = el.clientWidth;
      const hgt = el.clientHeight;
      renderer.setSize(w, hgt, false);
      renderer.domElement.style.width = "100%";
      renderer.domElement.style.height = "100%";
      camera.aspect = w / Math.max(1, hgt);
      camera.updateProjectionMatrix();
    };
    const ro = new ResizeObserver(resize);
    ro.observe(el);
    resize();
    let raf = 0;
    const loop = () => {
      controls.update();
      renderer.render(scene, camera);
      raf = requestAnimationFrame(loop);
    };
    loop();
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.dispose();
      renderer.dispose();
      el.removeChild(renderer.domElement);
      api.current = null;
    };
  }, [seg]);

  useEffect(() => {
    api.current?.setEdges(edges, litPath, hot);
  }, [edges, litPath, hot]);

  return <div ref={host} style={{ position: "absolute", inset: 0 }} aria-label="3D head with directed dependency arcs; drag to rotate" role="img" />;
}
