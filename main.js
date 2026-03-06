
'use strict';
const SERVER = "https://tests-8j3h.onrender.com";

// ═══════════════════════════════════════════
// AUDIO ENGINE (Web Audio API — no external)
// ═══════════════════════════════════════════
let audioCtx = null;
let soundEnabled = true;
let ambientNode = null;

function initAudio() {
  if (audioCtx) return;
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  startAmbient();
}

function startAmbient() {
  // Ambient drone disabled (removed noise)
}

function playClick() {
  if (!audioCtx || !soundEnabled) return;
  const osc = audioCtx.createOscillator();
  const g = audioCtx.createGain();
  osc.frequency.setValueAtTime(880, audioCtx.currentTime);
  osc.frequency.exponentialRampToValueAtTime(440, audioCtx.currentTime + 0.06);
  g.gain.setValueAtTime(0.18, audioCtx.currentTime);
  g.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.08);
  osc.connect(g); g.connect(audioCtx.destination);
  osc.start(); osc.stop(audioCtx.currentTime + 0.09);
}

function playPop(freq = 660) {
  if (!audioCtx || !soundEnabled) return;
  const osc = audioCtx.createOscillator();
  const g = audioCtx.createGain();
  osc.type = 'sine';
  osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
  osc.frequency.exponentialRampToValueAtTime(freq * 1.5, audioCtx.currentTime + 0.05);
  g.gain.setValueAtTime(0.12, audioCtx.currentTime);
  g.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.2);
  osc.connect(g); g.connect(audioCtx.destination);
  osc.start(); osc.stop(audioCtx.currentTime + 0.22);
}

function speakText(text) {
  if (!soundEnabled) return;
  if ('speechSynthesis' in window) {
    window.speechSynthesis.cancel();
    const utt = new SpeechSynthesisUtterance(text);
    utt.lang = 'ru-RU'; utt.rate = 0.95; utt.pitch = 1.25; utt.volume = 0.9;
    const voices = window.speechSynthesis.getVoices();
    const ru = voices.find(v => v.lang.startsWith('ru'));
    if (ru) utt.voice = ru;
    window.speechSynthesis.speak(utt);
  }
}

document.getElementById('soundBtn').addEventListener('click', () => {
  soundEnabled = !soundEnabled;
  document.getElementById('soundBtn').textContent = soundEnabled ? '🔊' : '🔇';
  if (ambientNode) ambientNode.gain.value = soundEnabled ? 0.08 : 0;
});

// Click sounds on chips and buttons
document.addEventListener('click', e => {
  if (e.target.classList.contains('chip') || e.target.closest('.go-btn') || e.target.closest('.w-btn')) {
    playClick();
  }
});

// ═══════════════════════════════════════════
// THREE.JS SETUP
// ═══════════════════════════════════════════
const canvas = document.getElementById('canvas3d');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.85;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x050810);
scene.fog = new THREE.FogExp2(0x050810, 0.035);

const camera = new THREE.PerspectiveCamera(58, window.innerWidth / window.innerHeight, 0.1, 80);
camera.position.set(0, 2.8, 7.5);
camera.lookAt(0, 1.5, 0);

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// ═══════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════
function mat(color, rough = 0.8, metal = 0) {
  return new THREE.MeshStandardMaterial({ color, roughness: rough, metalness: metal });
}
function neon(color, intensity = 1.2) {
  return new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: intensity, roughness: 0.1, metalness: 0.6 });
}
function makeSphere(r, col, rough = 0.85) {
  return new THREE.Mesh(new THREE.SphereGeometry(r, 24, 24), mat(col, rough));
}
// Capsule from two spheres + cylinder (no CapsuleGeometry!)
function makeCapsule(radius, height, color, roughness = 0.85) {
  const group = new THREE.Group();
  const m = mat(color, roughness);
  const cyl = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, height, 16), m.clone());
  const top = new THREE.Mesh(new THREE.SphereGeometry(radius, 16, 16), m.clone());
  const bot = new THREE.Mesh(new THREE.SphereGeometry(radius, 16, 16), m.clone());
  top.position.y = height / 2;
  bot.position.y = -height / 2;
  group.add(cyl, top, bot);
  return group;
}

// ═══════════════════════════════════════════
// LIGHTS
// ═══════════════════════════════════════════
scene.add(new THREE.AmbientLight(0x111830, 2.5));

const podLight = new THREE.PointLight(0x00e8ff, 10, 7);
podLight.position.set(0, 0.4, 0.3);
podLight.castShadow = true;
scene.add(podLight);

const keyLight = new THREE.DirectionalLight(0x334466, 2.5);
keyLight.position.set(2, 8, 5);
keyLight.castShadow = true;
keyLight.shadow.mapSize.set(1024, 1024);
scene.add(keyLight);

const rimL = new THREE.PointLight(0xff3fa4, 4, 9);
rimL.position.set(-4, 3, 2);
scene.add(rimL);
const rimR = new THREE.PointLight(0x4488ff, 3, 9);
rimR.position.set(4, 3, 2);
scene.add(rimR);

const ceilSpot = new THREE.SpotLight(0x00e8ff, 6, 14, Math.PI / 7, 0.6);
ceilSpot.position.set(0, 9.8, -0.5);
ceilSpot.target.position.set(0, 0, 0);
scene.add(ceilSpot, ceilSpot.target);

// ═══════════════════════════════════════════
// ROOM
// ═══════════════════════════════════════════
const wallM = mat(0x080c1a, 0.95);

// Floor
const floor = new THREE.Mesh(new THREE.PlaneGeometry(22, 18), mat(0x060a16, 0.85, 0.3));
floor.rotation.x = -Math.PI / 2;
floor.receiveShadow = true;
scene.add(floor);

// Floor reflection plane (fake, darker)
const floorRefl = new THREE.Mesh(new THREE.PlaneGeometry(22, 18),
  new THREE.MeshStandardMaterial({ color: 0x00e8ff, transparent: true, opacity: 0.03, roughness: 0.1, metalness: 1 }));
floorRefl.rotation.x = -Math.PI / 2;
floorRefl.position.y = 0.005;
scene.add(floorRefl);

scene.add(new THREE.GridHelper(22, 22, 0x00e8ff, 0x001a2e));

const backWall = new THREE.Mesh(new THREE.PlaneGeometry(22, 13), wallM);
backWall.position.set(0, 6.5, -8);
scene.add(backWall);
const leftWall = new THREE.Mesh(new THREE.PlaneGeometry(18, 13), wallM.clone());
leftWall.position.set(-9, 6.5, 0); leftWall.rotation.y = Math.PI / 2;
scene.add(leftWall);
const rightWall = new THREE.Mesh(new THREE.PlaneGeometry(18, 13), wallM.clone());
rightWall.position.set(9, 6.5, 0); rightWall.rotation.y = -Math.PI / 2;
scene.add(rightWall);
const ceiling = new THREE.Mesh(new THREE.PlaneGeometry(22, 18), wallM.clone());
ceiling.position.set(0, 11, -1); ceiling.rotation.x = Math.PI / 2;
scene.add(ceiling);

// Neon edge strips
function strip(x1,y1,z1, x2,y2,z2, color, opacity = 0.7) {
  const len = Math.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2);
  const geo = new THREE.CylinderGeometry(0.018, 0.018, len, 6);
  const m = new THREE.MeshBasicMaterial({ color, transparent: true, opacity });
  const mesh = new THREE.Mesh(geo, m);
  mesh.position.set((x1+x2)/2,(y1+y2)/2,(z1+z2)/2);
  if (Math.abs(y2-y1) < 0.01) mesh.rotation.z = Math.PI/2;
  else if (Math.abs(x2-x1) < 0.01 && Math.abs(z2-z1) > 0.01) {
    mesh.rotation.x = Math.PI/2;
  }
  scene.add(mesh);
}
strip(-9,0.02,-8, 9,0.02,-8, 0x00e8ff, 0.8);
strip(-9,0.02,5, 9,0.02,5, 0x00e8ff, 0.5);
strip(-9,0.02,-8, -9,0.02,5, 0x003344, 0.4);
strip(9,0.02,-8, 9,0.02,5, 0x003344, 0.4);
strip(-9,5,-8, 9,5,-8, 0x001a33, 0.3);

// Ceiling lamp
const lampM = neon(0x00e8ff, 2);
const lamp = new THREE.Mesh(new THREE.BoxGeometry(2.2, 0.07, 0.45), lampM);
lamp.position.set(0, 10.95, -0.5);
scene.add(lamp);
[0.9,-0.9].forEach(x => {
  const l2 = new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.06, 0.35), neon(0x00e8ff, 1.2));
  l2.position.set(x*3.5, 10.95, -0.5);
  scene.add(l2);
});

// Jungle window
const jungleM = new THREE.MeshBasicMaterial({ color: 0x002810 });
const jungleW = new THREE.Mesh(new THREE.PlaneGeometry(3.2, 2.4), jungleM);
jungleW.position.set(0, 4.8, -7.95);
scene.add(jungleW);
const jungleFrame = new THREE.LineSegments(
  new THREE.EdgesGeometry(new THREE.BoxGeometry(3.3, 2.5, 0.05)),
  new THREE.LineBasicMaterial({ color: 0x00e8ff }));
jungleFrame.position.copy(jungleW.position);
scene.add(jungleFrame);
// Bamboo inside window (green cylinders)
[-0.9,-0.3,0.3,0.9].forEach(x => {
  const b = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 2.2, 6),
    mat(0x1a5a20, 0.9));
  b.position.set(x, 4.8, -7.9);
  scene.add(b);
});

// Door
const doorM = mat(0x090c18, 0.9);
const door = new THREE.Mesh(new THREE.BoxGeometry(1.6, 3.2, 0.1), doorM);
door.position.set(-6.5, 1.6, -7.95);
scene.add(door);
const doorEdge = new THREE.LineSegments(
  new THREE.EdgesGeometry(new THREE.BoxGeometry(1.65, 3.25, 0.1)),
  new THREE.LineBasicMaterial({ color: 0x00e8ff }));
doorEdge.position.copy(door.position);
scene.add(doorEdge);
// "01" on door
const doorGlow = neon(0x00e8ff, 2);
const dNum = new THREE.Mesh(new THREE.BoxGeometry(0.6, 0.15, 0.05), doorGlow);
dNum.position.set(-6.5, 2.8, -7.88);
scene.add(dNum);

// Sofa
const sofaM = mat(0x181d2e, 0.95);
const sofaBase = new THREE.Mesh(new THREE.BoxGeometry(2.6, 0.7, 1.3), sofaM);
sofaBase.position.set(6, 0.35, -5);
scene.add(sofaBase);
const sofaBack = new THREE.Mesh(new THREE.BoxGeometry(2.6, 1.3, 0.3), mat(0x12172a, 0.95));
sofaBack.position.set(6, 1.05, -5.55);
scene.add(sofaBack);
// Sofa neon edge
const sofaEdge = neon(0x003355, 0.5);
const sofaStrip = new THREE.Mesh(new THREE.BoxGeometry(2.62, 0.04, 0.04), sofaEdge);
sofaStrip.position.set(6, 0.72, -4.35);
scene.add(sofaStrip);

// Control console left
const consM = mat(0x0a0e1a, 0.8, 0.5);
const cons = new THREE.Mesh(new THREE.BoxGeometry(1.5, 1.2, 0.8), consM);
cons.position.set(-6.5, 0.6, -5);
scene.add(cons);
const consScreen = new THREE.Mesh(new THREE.PlaneGeometry(1.1, 0.7), new THREE.MeshBasicMaterial({ color: 0x002244 }));
consScreen.position.set(-6.5, 0.85, -4.59);
scene.add(consScreen);
// Screen scan lines
for (let i = 0; i < 5; i++) {
  const sl = new THREE.Mesh(new THREE.BoxGeometry(0.9, 0.02, 0.01),
    neon(0x00e8ff, 0.8));
  sl.position.set(-6.5, 0.55 + i * 0.12, -4.58);
  scene.add(sl);
}

// ═══════════════════════════════════════════
// PODIUM
// ═══════════════════════════════════════════
const podium = new THREE.Mesh(new THREE.CylinderGeometry(1.5, 1.5, 0.06, 32),
  mat(0x060a16, 0.3, 0.9));
podium.receiveShadow = true;
scene.add(podium);
const ring1 = new THREE.Mesh(new THREE.TorusGeometry(1.45, 0.04, 8, 64), neon(0x00e8ff, 2));
ring1.rotation.x = Math.PI / 2; ring1.position.y = 0.035;
scene.add(ring1);
const ring2 = new THREE.Mesh(new THREE.TorusGeometry(1.75, 0.025, 8, 64), neon(0x003355, 0.8));
ring2.rotation.x = Math.PI / 2; ring2.position.y = 0.02;
scene.add(ring2);
// Pod glow plane
const podGlow = new THREE.Mesh(new THREE.CircleGeometry(1.4, 32),
  new THREE.MeshBasicMaterial({ color: 0x00e8ff, transparent: true, opacity: 0.06 }));
podGlow.rotation.x = -Math.PI / 2; podGlow.position.y = 0.01;
scene.add(podGlow);

// ═══════════════════════════════════════════
// PANDA (fixed: no CapsuleGeometry!)
// ═══════════════════════════════════════════
const pandaG = new THREE.Group();
scene.add(pandaG);

const wFur = mat(0xf2f2f2, 0.88);
const bFur = mat(0x111118, 0.9);
const dFur = mat(0x1e1e28, 0.9);
const pinkBlush = new THREE.MeshStandardMaterial({ color: 0xff9080, roughness: 0.9, transparent: true, opacity: 0.28 });
const cyN = neon(0x00e8ff, 1.4);

function addToP(mesh, shadow = true) {
  mesh.castShadow = shadow; pandaG.add(mesh); return mesh;
}

// BODY
const body = addToP(new THREE.Mesh(new THREE.SphereGeometry(0.9, 28, 28), wFur.clone()));
body.scale.set(1, 1.08, 0.95); body.position.set(0, 1.1, 0);

// Belly patch
const belly = addToP(new THREE.Mesh(new THREE.SphereGeometry(0.55, 20, 20),
  mat(0xfafafa, 0.88)));
belly.scale.set(1, 0.85, 0.65); belly.position.set(0, 1.0, 0.7);

// Circuit lines on body
const cLines = [
  [[-0.32,1.0,0.72],[0.32,1.0,0.72]],
  [[0,1.0,0.72],[0,0.68,0.72]],
  [[-0.26,0.84,0.72],[-0.26,1.0,0.72]],
  [[0.26,0.84,0.72],[0.26,1.0,0.72]],
];
cLines.forEach(([a,b]) => {
  const pts = [new THREE.Vector3(...a), new THREE.Vector3(...b)];
  const l = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
    new THREE.LineBasicMaterial({ color: 0x00e8ff }));
  pandaG.add(l);
});
const cDot = addToP(new THREE.Mesh(new THREE.SphereGeometry(0.045, 8, 8), cyN.clone()));
cDot.position.set(0, 0.7, 0.73);

// Chest exo plate
const plate = addToP(new THREE.Mesh(new THREE.BoxGeometry(0.55, 0.35, 0.04),
  mat(0x101828, 0.3, 0.9)));
plate.position.set(0, 1.0, 0.73);
const plateEdge = new THREE.LineSegments(
  new THREE.EdgesGeometry(new THREE.BoxGeometry(0.57, 0.37, 0.04)),
  new THREE.LineBasicMaterial({ color: 0x00e8ff }));
plateEdge.position.copy(plate.position); pandaG.add(plateEdge);

// HEAD
const head = addToP(new THREE.Mesh(new THREE.SphereGeometry(0.8, 32, 32), wFur.clone()));
head.position.set(0, 2.4, 0);
// Sheen
const sheen = addToP(new THREE.Mesh(new THREE.SphereGeometry(0.42, 16, 16),
  new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.8, transparent: true, opacity: 0.25 })));
sheen.scale.set(1, 0.55, 0.65); sheen.position.set(-0.16, 2.7, 0.42);

// Forehead chip
const chip = addToP(new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.1, 0.06), mat(0x080810, 0.3, 0.9)));
chip.position.set(0, 2.84, 0.72);
const chipG = addToP(new THREE.Mesh(new THREE.BoxGeometry(0.26, 0.12, 0.03), cyN.clone()));
chipG.position.set(0, 2.84, 0.74);
// Chip lines
for (let i = -1; i <= 1; i++) {
  const cl = new THREE.Mesh(new THREE.BoxGeometry(0.015, 0.1, 0.025), cyN.clone());
  cl.position.set(i * 0.07, 2.84, 0.74); pandaG.add(cl);
}

// EARS
function makeEar(sx) {
  const eg = new THREE.Group();
  const outer = new THREE.Mesh(new THREE.SphereGeometry(0.3, 18, 18), bFur.clone());
  outer.castShadow = true;
  const inner = new THREE.Mesh(new THREE.SphereGeometry(0.19, 14, 14), dFur.clone());
  inner.position.z = 0.1;
  const earRing = new THREE.Mesh(new THREE.TorusGeometry(0.11, 0.016, 6, 22), cyN.clone());
  earRing.position.z = 0.16;
  eg.add(outer, inner, earRing);
  eg.position.set(sx * 0.65, 2.98, 0.22);
  eg.rotation.z = sx * -0.32;
  eg.castShadow = true;
  pandaG.add(eg);
  return eg;
}
makeEar(-1); makeEar(1);

// EYE PATCHES
function makePatch(sx) {
  const p = new THREE.Mesh(new THREE.SphereGeometry(0.29, 18, 18), bFur.clone());
  p.scale.set(1, 0.82, 0.68); p.castShadow = true;
  p.position.set(sx * 0.33, 2.34, 0.65);
  p.rotation.z = sx * -0.14;
  pandaG.add(p);
}
makePatch(-1); makePatch(1);

// EYES (with changeable color!)
const eyeObjs = { left: {}, right: {} };
function makeEye(sx, side) {
  const base = new THREE.Mesh(new THREE.SphereGeometry(0.155, 18, 18),
    mat(0x001830, 0.1, 0.5));
  base.position.set(sx * 0.33, 2.35, 0.72);
  pandaG.add(base);

  const irisMat = new THREE.MeshStandardMaterial({ color: 0x00aad4, emissive: 0x00aad4, emissiveIntensity: 1.6, roughness: 0.1 });
  const iris = new THREE.Mesh(new THREE.SphereGeometry(0.108, 18, 18), irisMat);
  iris.position.set(sx * 0.33, 2.35, 0.8);
  pandaG.add(iris);

  const pupil = new THREE.Mesh(new THREE.SphereGeometry(0.058, 14, 14),
    new THREE.MeshBasicMaterial({ color: 0x000810 }));
  pupil.position.set(sx * 0.33, 2.35, 0.86);
  pandaG.add(pupil);

  const hl = new THREE.Mesh(new THREE.SphereGeometry(0.033, 8, 8),
    new THREE.MeshBasicMaterial({ color: 0xffffff }));
  hl.position.set(sx * 0.34, 2.375, 0.895);
  pandaG.add(hl);

  const eyeRingMat = cyN.clone();
  const eyeRing = new THREE.Mesh(new THREE.TorusGeometry(0.138, 0.009, 6, 22), eyeRingMat);
  eyeRing.position.set(sx * 0.33, 2.35, 0.71);
  pandaG.add(eyeRing);

  eyeObjs[side] = { iris: irisMat, ring: eyeRingMat, eyeRingMesh: eyeRing };
}
makeEye(-1, 'left'); makeEye(1, 'right');

// Blush
[-1, 1].forEach(sx => {
  const b = new THREE.Mesh(new THREE.SphereGeometry(0.19, 12, 12), pinkBlush.clone());
  b.scale.set(1, 0.48, 0.45); b.position.set(sx * 0.52, 2.27, 0.62); pandaG.add(b);
});

// Nose
const nose = new THREE.Mesh(new THREE.SphereGeometry(0.1, 14, 14), mat(0x0a0a12, 0.9));
nose.scale.set(1, 0.62, 0.65); nose.position.set(0, 2.19, 0.78); pandaG.add(nose);

// Side tech dots
[-0.88, 0.88].forEach(x => {
  const d = new THREE.Mesh(new THREE.SphereGeometry(0.032, 8, 8), cyN.clone());
  d.position.set(x, 2.35, 0.32); pandaG.add(d);
  const pts = [new THREE.Vector3(x, 2.35, 0.32), new THREE.Vector3(x * 0.6, 2.35, 0.67)];
  pandaG.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
    new THREE.LineBasicMaterial({ color: 0x00e8ff, transparent: true, opacity: 0.5 })));
});

// LEFT ARM (capsule: cylinder + 2 spheres)
const leftArmG = new THREE.Group();
const lC = new THREE.Mesh(new THREE.CylinderGeometry(0.19, 0.16, 0.55, 14), wFur.clone());
const lS1 = new THREE.Mesh(new THREE.SphereGeometry(0.19, 14, 14), wFur.clone());
const lS2 = new THREE.Mesh(new THREE.SphereGeometry(0.16, 14, 14), wFur.clone());
lS1.position.y = 0.275; lS2.position.y = -0.275;
leftArmG.add(lC, lS1, lS2);
const lPaw = new THREE.Mesh(new THREE.SphereGeometry(0.19, 14, 14), bFur.clone());
lPaw.position.set(0, -0.52, 0.12);
leftArmG.add(lPaw);
leftArmG.position.set(-0.92, 1.52, 0.05);
leftArmG.rotation.z = 0.38;
[lC,lS1,lS2,lPaw].forEach(m => { m.castShadow = true; });
pandaG.add(leftArmG);

// RIGHT ARM — anatomically correct waving arm
// Structure: shoulder joint → upper arm → elbow → forearm → wrist → paw
// Pivot at SHOULDER so entire arm waves naturally
const rightArmG = new THREE.Group();
rightArmG.position.set(0.88, 1.72, 0.12); // shoulder anchor on body

// Shoulder ball joint (connects arm to body, hides seam)
const rShoulder = new THREE.Mesh(new THREE.SphereGeometry(0.22, 16, 16), wFur.clone());
rShoulder.position.set(0, 0, 0);
rShoulder.castShadow = true;
rightArmG.add(rShoulder);

// Upper arm — cylinder tapers from shoulder to elbow
const rUpperArm = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.13, 0.46, 14), wFur.clone());
rUpperArm.position.set(0.08, 0.28, 0.05);
rUpperArm.rotation.z = -0.18;
rUpperArm.castShadow = true;
rightArmG.add(rUpperArm);

// Elbow joint — slightly darker, smaller sphere for definition
const rElbow = new THREE.Mesh(new THREE.SphereGeometry(0.14, 14, 14), mat(0xe0e0e0, 0.9));
rElbow.position.set(0.12, 0.52, 0.08);
rElbow.castShadow = true;
rightArmG.add(rElbow);

// Forearm — thinner, angled upward for wave pose
const rForeArm = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.14, 0.4, 14), wFur.clone());
rForeArm.position.set(0.16, 0.76, 0.1);
rForeArm.rotation.z = -0.55;
rForeArm.rotation.x = 0.15;
rForeArm.castShadow = true;
rightArmG.add(rForeArm);

// Wrist joint
const rWrist = new THREE.Mesh(new THREE.SphereGeometry(0.115, 12, 12), wFur.clone());
rWrist.position.set(0.24, 0.96, 0.14);
rWrist.castShadow = true;
rightArmG.add(rWrist);

// PAW — panda has rounded paw, not human hand
// Base of paw: flattened sphere
const rPawBase = new THREE.Mesh(new THREE.SphereGeometry(0.18, 16, 16), bFur.clone());
rPawBase.scale.set(1.1, 0.7, 1.0);
rPawBase.position.set(0.28, 1.08, 0.17);
rPawBase.castShadow = true;
rightArmG.add(rPawBase);

// Paw pads — 1 large central + 4 small toe pads (NOT fingers)
const padMat = mat(0x2a2030, 0.85);
// Central palm pad
const palmPad = new THREE.Mesh(new THREE.SphereGeometry(0.08, 10, 10), padMat.clone());
palmPad.scale.set(1, 0.4, 1);
palmPad.position.set(0.28, 1.14, 0.24);
rightArmG.add(palmPad);
// 4 small toe pads arranged in arc (panda has 5 toes but visually 4 visible)
const toeOffsets = [
  [-0.07, 0.02], [-0.025, 0.06], [0.03, 0.065], [0.08, 0.035]
];
toeOffsets.forEach(([dx, dz]) => {
  const toe = new THREE.Mesh(new THREE.SphereGeometry(0.038, 8, 8), padMat.clone());
  toe.scale.set(1, 0.45, 1);
  toe.position.set(0.28 + dx, 1.18, 0.25 + dz);
  rightArmG.add(toe);
});

// Circuit line on forearm (cyber detail)
const armCircDot = new THREE.Mesh(new THREE.SphereGeometry(0.024, 8, 8), cyN.clone());
armCircDot.position.set(0.14, 0.72, 0.16);
rightArmG.add(armCircDot);
const armWirePts = [
  new THREE.Vector3(0.1, 0.6, 0.12),
  new THREE.Vector3(0.14, 0.72, 0.16),
  new THREE.Vector3(0.18, 0.85, 0.14)
];
rightArmG.add(new THREE.Line(
  new THREE.BufferGeometry().setFromPoints(armWirePts),
  new THREE.LineBasicMaterial({ color: 0x00e8ff, transparent: true, opacity: 0.7 })
));

// Shadow-casting dark circle under arm where it meets body (hides seam)
const armShadowMat = new THREE.MeshBasicMaterial({ color: 0x080810, transparent: true, opacity: 0.35 });
const armShadow = new THREE.Mesh(new THREE.CircleGeometry(0.2, 16), armShadowMat);
armShadow.position.set(-0.06, 1.72, 0.18);
armShadow.rotation.y = -0.3;
pandaG.add(armShadow); // on pandaG not armG so it stays put

// Initial rotation: arm raised for waving
rightArmG.rotation.z = -1.1;  // arm raised outward
rightArmG.rotation.x = 0.2;   // slightly forward
pandaG.add(rightArmG);

// LEGS
[-1, 1].forEach(sx => {
  const lg = new THREE.Group();
  const uL = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.22, 0.45, 14), wFur.clone());
  const uS1 = new THREE.Mesh(new THREE.SphereGeometry(0.2, 14, 14), wFur.clone());
  const uS2 = new THREE.Mesh(new THREE.SphereGeometry(0.22, 14, 14), wFur.clone());
  uS1.position.y = 0.225; uS2.position.y = -0.225;
  const foot = new THREE.Mesh(new THREE.SphereGeometry(0.21, 14, 14), bFur.clone());
  foot.scale.set(1.22, 0.65, 1.3); foot.position.set(0, -0.46, 0.1);
  [uL,uS1,uS2,foot].forEach(m => { m.castShadow = true; lg.add(m); });
  lg.position.set(sx * 0.42, 0.55, 0.12);
  pandaG.add(lg);
});

// TAIL
const tail = new THREE.Mesh(new THREE.SphereGeometry(0.19, 14, 14), wFur.clone());
tail.position.set(0, 1.02, -0.85); pandaG.add(tail);

// ═══════════════════════════════════════════
// NEON 3D CUBES
// ═══════════════════════════════════════════
const cubeData = [
  { color: 0x00e8ff, label: 'AI',   px: -2.2, py: 1.2, pz: -1 },
  { color: 0xff3fa4, label: 'DATA', px:  2.2, py: 1.2, pz: -1 },
  { color: 0xb8ff2e, label: 'SYS',  px: -2.5, py: 0.6, pz:  1 },
  { color: 0xff8c00, label: 'NET',  px:  2.5, py: 0.6, pz:  1 },
  { color: 0xa855f7, label: 'MEM',  px: -1.9, py: 2.3, pz:  0.5 },
  { color: 0x00ff88, label: 'CPU',  px:  1.9, py: 2.3, pz:  0.5 },
];

const cubeObjs = [];
const raycaster = new THREE.Raycaster();
const mouse2 = new THREE.Vector2(-9999, -9999);
let dragCube = null, dragPlane = new THREE.Plane(), dragOffset = new THREE.Vector3();

cubeData.forEach(d => {
  const g = new THREE.Group();
  g.position.set(d.px, d.py, d.pz);
  g.userData = { ...d, homeY: d.py, isDraggable: true };

  const boxGeo = new THREE.BoxGeometry(0.48, 0.48, 0.48);
  g.add(new THREE.LineSegments(new THREE.EdgesGeometry(boxGeo),
    new THREE.LineBasicMaterial({ color: d.color })));
  g.add(new THREE.Mesh(boxGeo,
    new THREE.MeshBasicMaterial({ color: d.color, transparent: true, opacity: 0.07, side: THREE.DoubleSide })));

  const core = new THREE.Mesh(new THREE.SphereGeometry(0.1, 8, 8),
    new THREE.MeshBasicMaterial({ color: d.color, transparent: true, opacity: 0.5 }));
  g.add(core);

  const pl = new THREE.PointLight(d.color, 1.8, 3.5);
  g.add(pl);

  scene.add(g);
  cubeObjs.push(g);
});

// Wires between cubes and to panda center
const wireGeos = [];
function buildWires() {
  wireGeos.forEach(({ line }) => scene.remove(line));
  wireGeos.length = 0;
  cubeObjs.forEach((c, i) => {
    const next = cubeObjs[(i + 1) % cubeObjs.length];
    addWire(c.position, next.position, c.userData.color, 0.2);
    addWire(c.position, new THREE.Vector3(0, 1.8, 0), c.userData.color, 0.12);
  });
}
function addWire(a, b, color, opacity) {
  const pts = [a.clone(), b.clone()];
  const geo = new THREE.BufferGeometry().setFromPoints(pts);
  const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color, transparent: true, opacity }));
  scene.add(line);
  wireGeos.push({ line, geo });
}
buildWires();

function updateWires() {
  // Rebuild instead of update for simplicity
  buildWires();
}

// DRAG
canvas.addEventListener('mousedown', e => {
  updateMouseVec(e.clientX, e.clientY);
  raycaster.setFromCamera(mouse2, camera);
  const hits = raycaster.intersectObjects(cubeObjs, true);
  if (hits.length) {
    let obj = hits[0].object;
    while (obj.parent && !obj.userData.isDraggable) obj = obj.parent;
    if (obj.userData.isDraggable) {
      dragCube = obj;
      dragPlane.setFromNormalAndCoplanarPoint(
        camera.getWorldDirection(new THREE.Vector3()).negate(), dragCube.position);
      const pt = new THREE.Vector3();
      raycaster.ray.intersectPlane(dragPlane, pt);
      dragOffset.copy(pt).sub(dragCube.position);
      canvas.style.cursor = 'grabbing';
      playClick();
    }
  }
});
canvas.addEventListener('mousemove', e => {
  updateMouseVec(e.clientX, e.clientY);
  if (dragCube) {
    raycaster.setFromCamera(mouse2, camera);
    const pt = new THREE.Vector3();
    raycaster.ray.intersectPlane(dragPlane, pt);
    dragCube.position.copy(pt.sub(dragOffset));
    dragCube.position.y = Math.max(0.35, dragCube.position.y);
  } else {
    raycaster.setFromCamera(mouse2, camera);
    const h = raycaster.intersectObjects(cubeObjs, true);
    canvas.style.cursor = h.length ? 'grab' : 'default';
  }
});
canvas.addEventListener('mouseup', () => {
  dragCube = null; canvas.style.cursor = 'default';
});
function updateMouseVec(cx, cy) {
  const r = canvas.getBoundingClientRect();
  mouse2.set(((cx - r.left) / r.width) * 2 - 1, -((cy - r.top) / r.height) * 2 + 1);
}

// ═══════════════════════════════════════════
// PARTICLES (3D, burst toward panda)
// ═══════════════════════════════════════════
const parts = [];

function spawnBurst(emotion, fromUI = false) {
  const COLS = {
    happy: [0x00e8ff, 0xffffff, 0xffee00],
    excited: [0xffaa00, 0xff4400, 0xffff00],
    love: [0xff3fa4, 0xff88cc, 0xff0055],
    thinking: [0x4488ff, 0x00e8ff, 0xaaaaff],
    surprised: [0xff8c00, 0xffff00, 0xffffff],
    angry: [0xff2200, 0xff8800, 0xfff000],
    sleepy: [0x224488, 0x4466cc, 0x00e8ff],
  };
  const cols = COLS[emotion] || COLS.happy;
  const count = fromUI ? 20 : 14;

  for (let i = 0; i < count; i++) {
    const color = cols[Math.floor(Math.random() * cols.length)];
    const size = 0.04 + Math.random() * 0.07;
    const mesh = new THREE.Mesh(
      new THREE.SphereGeometry(size, 6, 6),
      new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 1 })
    );
    const angle = Math.random() * Math.PI * 2;
    const r = fromUI ? 3 + Math.random() : 0.6 + Math.random() * 0.9;
    mesh.position.set(
      Math.cos(angle) * r,
      fromUI ? 0.5 + Math.random() * 2 : 1.2 + Math.random() * 1.8,
      Math.sin(angle) * r * 0.4
    );
    // Velocity toward panda center
    const target = new THREE.Vector3(0, 1.8, 0);
    const vel = target.clone().sub(mesh.position).normalize().multiplyScalar(0.04 + Math.random() * 0.03);
    vel.x += (Math.random() - 0.5) * 0.05;
    vel.y += (Math.random() - 0.5) * 0.04;

    mesh.userData.vel = vel;
    mesh.userData.life = 1;
    mesh.userData.decay = 0.014 + Math.random() * 0.016;
    mesh.userData.spin = (Math.random() - 0.5) * 0.12;
    scene.add(mesh);
    parts.push(mesh);
  }
  playPop(440 + Math.random() * 220);
}

// ═══════════════════════════════════════════
// EMOTION + EYE COLOR CHANGE
// ═══════════════════════════════════════════
let curEmotion = 'happy';
const EYE_COLORS = {
  happy:    { iris: 0x00aad4, ring: 0x00e8ff, pod: 0x00e8ff, rim: 0xff3fa4 },
  excited:  { iris: 0xffcc00, ring: 0xffaa00, pod: 0xffaa00, rim: 0xffaa00 },
  love:     { iris: 0xff3fa4, ring: 0xff3fa4, pod: 0xff3fa4, rim: 0xff3fa4 },
  thinking: { iris: 0x4488ff, ring: 0x4488ff, pod: 0x4488ff, rim: 0x4488ff },
  surprised:{ iris: 0xffffff, ring: 0xff8c00, pod: 0xff8800, rim: 0xff8c00 },
  angry:    { iris: 0xff2200, ring: 0xff2200, pod: 0xff0000, rim: 0xff2200 },
  sleepy:   { iris: 0x224488, ring: 0x003366, pod: 0x002244, rim: 0x224488 },
};
const EMO_BADGES = {
  happy:    '◈ СЧАСТЛИВЫЙ',
  excited:  '◈ ВОСТОРГ!',
  love:     '◈ ЛЮБОВЬ',
  thinking: '◈ АНАЛИЗ...',
  surprised:'◈ УДИВЛЕНИЕ!',
  angry:    '◈ СЕРДИТСЯ!',
  sleepy:   '◈ ОЖИДАНИЕ...',
};

function setEmotion(e) {
  curEmotion = e;
  const c = EYE_COLORS[e] || EYE_COLORS.happy;
  // Change eye iris color
  ['left','right'].forEach(side => {
    const o = eyeObjs[side];
    if (!o.iris) return;
    o.iris.color.setHex(c.iris);
    o.iris.emissive.setHex(c.iris);
    o.ring.color.setHex(c.ring);
    o.ring.emissive.setHex(c.ring);
  });
  podLight.color.setHex(c.pod);
  rimL.color.setHex(c.rim);
  // Update badge
  const badge = document.getElementById('emoBadge');
  badge.textContent = EMO_BADGES[e] || '◈ ...';
  badge.style.color = '#' + c.ring.toString(16).padStart(6,'0');
  badge.style.textShadow = `0 0 10px #${c.ring.toString(16).padStart(6,'0')}`;
  // Chat bubble border
  const bbl = document.getElementById('chatBubble');
  bbl.className = 'glass';
  if (e === 'thinking') bbl.classList.add('thinking');
  else if (e === 'angry') bbl.classList.add('angry');
  else if (e === 'love') bbl.classList.add('love');
  spawnBurst(e);
}

// ═══════════════════════════════════════════
// MOUSE PARALLAX
// ═══════════════════════════════════════════
let mxN = 0, myN = 0;
document.addEventListener('mousemove', e => {
  mxN = (e.clientX / window.innerWidth - 0.5) * 2;
  myN = (e.clientY / window.innerHeight - 0.5) * 2;
});

// ═══════════════════════════════════════════
// ANIMATION LOOP
// ═══════════════════════════════════════════
const clock = new THREE.Clock();

function animate() {
  requestAnimationFrame(animate);
  const t = clock.getElapsedTime();

  // PANDA FLOAT + BREATHE
  pandaG.position.y = Math.sin(t * 1.1) * 0.09;
  body.scale.set(1 + Math.sin(t * 1.6) * 0.012, 1.08 + Math.sin(t * 1.6) * 0.01, 0.95);

  // PANDA LOOK AT MOUSE (parallax)
  pandaG.rotation.y += (mxN * 0.28 - pandaG.rotation.y) * 0.04;
  head.rotation.y += (mxN * 0.14 - head.rotation.y) * 0.05;
  head.rotation.x += (-myN * 0.1 - head.rotation.x) * 0.05;

  // EMOTION-BASED ANIMATIONS
  if (curEmotion === 'happy' || curEmotion === 'excited') {
    // Wave: rotate entire arm from SHOULDER pivot
    // Up-down wave on Z axis, slight forearm twist on X
    const wave = Math.sin(t * 3.6);
    rightArmG.rotation.z = -1.1 + wave * 0.45;   // shoulder raises/lowers
    rightArmG.rotation.x = 0.2 + Math.abs(wave) * 0.2; // slight forward rock
    rightArmG.rotation.y = wave * 0.12;            // subtle twist
    leftArmG.rotation.z = 0.38 + Math.sin(t * 1.4) * 0.05;
    tail.position.x = 0;
  } else if (curEmotion === 'angry') {
    // Fists at sides, tense
    rightArmG.rotation.z = -0.55 + Math.sin(t * 5.5) * 0.18;
    rightArmG.rotation.x = 0.1;
    rightArmG.rotation.y = Math.sin(t * 5.5) * 0.08;
    leftArmG.rotation.z = 0.55 + Math.sin(t * 5.5 + 1) * 0.18;
    tail.position.x = Math.sin(t * 8.5) * 0.2;
    tail.position.z = -0.85 + Math.cos(t * 8.5) * 0.06;
  } else if (curEmotion === 'thinking') {
    // Right arm raised to chin
    rightArmG.rotation.z = -1.35;
    rightArmG.rotation.x = 0.55;
    rightArmG.rotation.y = -0.2;
    leftArmG.rotation.z = 0.38;
    tail.position.x = 0;
  } else if (curEmotion === 'love') {
    // Both arms slightly open
    rightArmG.rotation.z = -0.85 + Math.sin(t * 1.8) * 0.15;
    rightArmG.rotation.x = 0.25;
    leftArmG.rotation.z = 0.85 + Math.sin(t * 1.8 + 1) * 0.15;
    tail.position.x = 0;
  } else if (curEmotion === 'sleepy') {
    // Arms drooping
    rightArmG.rotation.z = -0.3;
    rightArmG.rotation.x = 0.05;
    leftArmG.rotation.z = 0.3;
    head.rotation.x = 0.12 + Math.sin(t * 0.5) * 0.06;
  } else {
    rightArmG.rotation.z = -0.52 + Math.sin(t * 1.4) * 0.08;
    leftArmG.rotation.z = 0.38 + Math.sin(t * 1.4) * 0.05;
    tail.position.x = 0;
  }

  // BLINK
  const bt = t % 4.5;
  const blinkY = (bt > 4.2 && bt < 4.38) ? 0.15 : 1;
  // Eye ring rotation
  Object.values(eyeObjs).forEach(o => {
    if (o.eyeRingMesh) o.eyeRingMesh.rotation.z = t * 0.6;
  });

  // POD LIGHT PULSE
  podLight.intensity = 8 + Math.sin(t * 2.2) * 2.5;

  // RINGS
  ring1.rotation.z = t * 0.55;
  ring2.rotation.z = -t * 0.32;
  ring2.material.opacity = 0.5 + Math.sin(t * 1.8) * 0.2;

  // CUBES
  cubeObjs.forEach((c, i) => {
    if (dragCube !== c) {
      c.rotation.x = t * 0.55 + i * 1.1;
      c.rotation.y = t * 0.7 + i * 0.9;
      c.position.y = c.userData.homeY + Math.sin(t * 1.4 + i * 1.1) * 0.14;
    }
  });

  // PARTICLES
  for (let i = parts.length - 1; i >= 0; i--) {
    const p = parts[i];
    p.userData.life -= p.userData.decay;
    p.material.opacity = p.userData.life;
    p.position.addScaledVector(p.userData.vel, 1);
    p.rotation.z += p.userData.spin;
    if (p.userData.life <= 0) { scene.remove(p); parts.splice(i, 1); }
  }

  renderer.render(scene, camera);
}
animate();

// ═══════════════════════════════════════════
// UI EVENTS
// ═══════════════════════════════════════════
let inChat = false;

window.addEventListener('load', () => {
  setTimeout(() => {
    document.getElementById('splash').classList.add('hide');
    spawnBurst('happy');
  }, 3800);
});

document.getElementById('startBtn').addEventListener('click', () => {
  initAudio();
  document.getElementById('welcomeUI').classList.add('hide');
  document.getElementById('chatUI').style.display = 'block';
  inChat = true;
  playPop(660);
  spawnBurst('excited', true);
  // Smooth camera move
  const target = { z: 6.5, y: 2.4 };
  const step = () => {
    camera.position.z += (target.z - camera.position.z) * 0.04;
    camera.position.y += (target.y - camera.position.y) * 0.04;
    if (Math.abs(camera.position.z - target.z) > 0.02) requestAnimationFrame(step);
  };
  step();
});

document.getElementById('backBtn').addEventListener('click', () => {
  document.getElementById('chatUI').style.display = 'none';
  document.getElementById('welcomeUI').classList.remove('hide');
  inChat = false;
  setEmotion('happy');
  chatHistory.length = 0; // сброс истории при возврате в меню
});

// ═══════════════════════════════════════════
// ROOM INTERACTIVITY
// ═══════════════════════════════════════════
let skinPanelOpen = false;
function toggleSkinPanel() {
  skinPanelOpen = !skinPanelOpen;
  document.getElementById('skinPanel').style.display = skinPanelOpen ? 'block' : 'none';
  playClick();
}
function setSkin(skin) {
  activeSkin = skin;
  document.querySelectorAll('.skin-opt').forEach(el => el.classList.remove('active'));
  event.target.classList.add('active');
  const skinColors = {
    default:{ fur:0xf2f2f2, dark:0x111118 },
    cyber:  { fur:0xd0e8ff, dark:0x0a1830 },
    ninja:  { fur:0x222222, dark:0x050505 },
    astro:  { fur:0xfff0cc, dark:0x1a0e00 },
  };
  const c = skinColors[skin] || skinColors.default;
  pandaG.traverse(obj => {
    if (!obj.isMesh || !obj.material?.color) return;
    const h = obj.material.color.getHex();
    if ([0xf2f2f2,0xd0e8ff,0x222222,0xfff0cc,0xe0e0e0,0xfafafa].includes(h))
      obj.material.color.setHex(c.fur);
    else if ([0x111118,0x0a1830,0x050505,0x1a0e00,0x181820,0x1e1e28,0x2a2030].includes(h))
      obj.material.color.setHex(c.dark);
  });
  playPop(550); spawnBurst('excited', true);
  skinPanelOpen = false;
  document.getElementById('skinPanel').style.display = 'none';
}
let activeSkin = 'default';

// Memory frame
const memTopics = [
  {icon:'🧠',label:'ПАМЯТЬ'},{icon:'🎮',label:'ИГРЫ'},
  {icon:'🎵',label:'МУЗЫКА'},{icon:'📐',label:'МАТЕМ'},
  {icon:'🌌',label:'КОСМОС'},{icon:'🤖',label:'AI'},{icon:'🧪',label:'НАУКА'},
];
let memIdx = 0;
function cycleMemory() { memIdx=(memIdx+1)%memTopics.length; updateMemIcon(); playClick(); }
function updateMemIcon() {
  document.getElementById('memIcon').textContent = memTopics[memIdx].icon;
  document.getElementById('memLabel').textContent = memTopics[memIdx].label;
}
function updateMemory(text) {
  const t = text.toLowerCase();
  if (t.includes('игр')||t.includes('game')) memIdx=1;
  else if (t.includes('муз')||t.includes('music')) memIdx=2;
  else if (t.includes('мат')||t.includes('числ')) memIdx=3;
  else if (t.includes('космос')||t.includes('space')) memIdx=4;
  else if (t.includes('ai')||t.includes('робот')) memIdx=5;
  else if (t.includes('наук')||t.includes('химия')) memIdx=6;
  updateMemIcon();
}

// Neon pulse
function triggerNeonPulse() {
  const el=document.getElementById('neonPulse');
  el.classList.remove('active'); void el.offsetWidth; el.classList.add('active');
}

// Terminal mini-game
const CODE_CHARS='ABCDEFGHJKLMNPQRSTUVWXYZ0123456789';
let tgScore=0, tgTimer=null, tgCode='';
function openTermGame() {
  document.getElementById('termGame').classList.add('open');
  tgNextCode(); document.getElementById('tgInput').focus(); playClick();
}
function closeTermGame() {
  document.getElementById('termGame').classList.remove('open');
  clearTimeout(tgTimer); document.getElementById('tgInput').value='';
}
function tgNextCode() {
  tgCode=Array.from({length:4},()=>CODE_CHARS[Math.floor(Math.random()*CODE_CHARS.length)]).join('');
  document.getElementById('tgTarget').textContent=tgCode;
  document.getElementById('tgInput').value='';
  document.getElementById('tgMsg').textContent='';
  clearTimeout(tgTimer);
  tgTimer=setTimeout(()=>{
    document.getElementById('tgMsg').textContent='✗ ВРЕМЯ ВЫШЛО';
    document.getElementById('tgMsg').style.color='#ff3333';
    setTimeout(tgNextCode,900);
  },3500);
}
document.addEventListener('input',e=>{
  if(e.target.id!=='tgInput') return;
  const val=e.target.value.toUpperCase(); e.target.value=val;
  if(val===tgCode){
    tgScore++;
    document.getElementById('tgScore').textContent=`⚡ ЭНЕРГИЯ: ${tgScore}`;
    document.getElementById('tgMsg').textContent='✓ ВЗЛОМАНО!';
    document.getElementById('tgMsg').style.color='#b8ff2e';
    playPop(880); clearTimeout(tgTimer); setTimeout(tgNextCode,600);
  }
});

// ═══════════════════════════════════════════
// CHAT + ИСТОРИЯ ДИАЛОГА
// ═══════════════════════════════════════════

// История хранится в памяти браузера на время сессии
// Формат: [{role:'user', content:'...'}, {role:'assistant', content:'...'}]
const chatHistory = [];
const MAX_HISTORY_PAIRS = 10; // максимум пар user+assistant

async function send() {
  const inp = document.getElementById('mIn');
  const t = inp.value.trim(); if (!t) return;
  updateMemory(t); inp.value = ''; await ask(t);
}
async function sq(t) { spawnBurst('excited',true); updateMemory(t); await ask(t); }

async function ask(text) {
  const btn = document.getElementById('gBtn');
  const chatText = document.getElementById('chatText');
  btn.textContent = '⏳'; btn.disabled = true;
  setEmotion('thinking'); triggerNeonPulse();
  chatText.innerHTML = 'Думаю...<span style="display:inline-block;width:7px;height:13px;background:#4488ff;margin-left:4px;vertical-align:middle;animation:blink 0.7s infinite"></span>';
  try {
    // Обрезаем историю до MAX_HISTORY_PAIRS пар перед отправкой
    const historyToSend = chatHistory.slice(-(MAX_HISTORY_PAIRS * 2));

    const r = await fetch(SERVER+'/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        message: text,
        history: historyToSend   // ← передаём историю на сервер
      })
    });
    const d = await r.json();

    // Сохраняем пару user+assistant в историю
    chatHistory.push({ role: 'user',      content: text });
    chatHistory.push({ role: 'assistant', content: d.text });

    chatText.textContent = d.text;
    setEmotion(d.emotion||'happy'); triggerNeonPulse();
    setTimeout(()=>speakText(d.text),300);
  } catch(err) {
    chatText.textContent='Сервер спит... Подожди 30 сек 😴'; setEmotion('sleepy');
  }
  btn.textContent='▶'; btn.disabled=false;
}

