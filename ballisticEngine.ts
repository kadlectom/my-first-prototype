// ============================================
// BCT Ballistic Engine
// Point Mass Model with G1 Drag
// ============================================

export type ClickUnit = "0.1MIL" | "0.05MIL" | "1/4MOA" | "1/8MOA";

export interface Profile {
  name: string;
  caliber: string;
  bulletType: string;
  bulletMass: number;
  massUnit: "g" | "gr";
  bcG1: number;
  muzzleVelocity: number;  // m/s
  scopeHeight: number;     // mm
  zeroRange: number;       // m
  clickUnit: ClickUnit;
}

export interface Environment {
  temperature: number;  // °C
  pressure: number;     // hPa
  humidity: number;     // %
  altitude: number;     // m
}

export interface Wind {
  speed: number;           // m/s
  directionClock: number;  // 1–12
}

export interface ShotInput {
  distance: number;       // m
  shootingAngle: number;  // degrees, positive = uphill
}

export interface BallisticResult {
  elevationMrad: number;
  elevationClicks: number;
  windageMrad: number;
  windageClicks: number;
  dropCm: number;
  driftCm: number;
}

export interface TrajectoryPoint {
  distance: number;
  dropCm: number;
  elevationMrad: number;
  elevationClicks: number;
  windDriftCm: number;
  windMrad: number;
  windClicks: number;
  velocity: number;
  energy: number;
}

// ============================================
// Constants
// ============================================

const GRAVITY = 9.80665;
const R_AIR = 287.05;
const BC_CONVERSION = 703.07; // 1 lb/in² = 703.07 kg/m²
const SPIN_DRIFT_K = 0.00012;
const MAX_DISTANCE = 3000;
const MIN_ENERGY = 100;
const TRAJECTORY_STEP = 25;

// ============================================
// G1 Drag Table (corrected)
// Cd peaks in transonic region (~Mach 1.1) then decreases supersonically.
// ============================================

interface DragPoint {
  mach: number;
  cd: number;
}

const G1_TABLE: DragPoint[] = [
  { mach: 0.00, cd: 0.2629 },
  { mach: 0.50, cd: 0.2032 },
  { mach: 0.70, cd: 0.2166 },
  { mach: 0.90, cd: 0.3125 },
  { mach: 1.00, cd: 0.4310 },
  { mach: 1.10, cd: 0.4950 },  // peak
  { mach: 1.20, cd: 0.4820 },
  { mach: 1.30, cd: 0.4640 },
  { mach: 1.40, cd: 0.4440 },
  { mach: 1.50, cd: 0.4240 },
  { mach: 1.60, cd: 0.4040 },
  { mach: 1.80, cd: 0.3680 },
  { mach: 2.00, cd: 0.3350 },
  { mach: 2.50, cd: 0.2850 },
  { mach: 3.00, cd: 0.2490 },
];

// ============================================
// Utility Functions
// ============================================

function airDensity(env: Environment): number {
  const T = env.temperature + 273.15;  // K
  const p = env.pressure * 100;        // hPa → Pa
  return p / (R_AIR * T);
}

function speedOfSound(tempC: number): number {
  return 331.3 + 0.606 * tempC;  // m/s
}

// FIX: handle both grams and grains
function bulletMassKg(profile: Profile): number {
  return profile.massUnit === "gr"
    ? profile.bulletMass * 0.00006479891
    : profile.bulletMass / 1000;
}

function interpolateDrag(mach: number): number {
  if (mach <= G1_TABLE[0].mach) return G1_TABLE[0].cd;
  for (let i = 0; i < G1_TABLE.length - 1; i++) {
    const a = G1_TABLE[i];
    const b = G1_TABLE[i + 1];
    if (mach >= a.mach && mach <= b.mach) {
      const t = (mach - a.mach) / (b.mach - a.mach);
      return a.cd + t * (b.cd - a.cd);
    }
  }
  return G1_TABLE[G1_TABLE.length - 1].cd;
}

// Wind cross-component perpendicular to shot direction (positive = right drift)
// Clock system: 12 = headwind (0 cross), 3 = full right, 6 = tailwind (0 cross), 9 = full left
function windCrossComponent(wind: Wind): number {
  return wind.speed * Math.sin((wind.directionClock / 6) * Math.PI);
}

// ============================================
// Drag Acceleration
// FIX: include atmospheric density correction (rho / rho0)
// ============================================

function dragAcceleration(
  velocity: number,
  bc: number,
  mach: number,
  rho: number
): number {
  const cd = interpolateDrag(mach);
  const bcSI = bc * BC_CONVERSION;  // convert lb/in² → kg/m²
  return 0.5 * rho * cd * velocity * velocity / bcSI;
}

// ============================================
// RK4 Integration Step
// ============================================

function rk4Step(
  state: number[],
  dt: number,
  deriv: (s: number[]) => number[]
): number[] {
  const k1 = deriv(state);
  const s2 = state.map((v, i) => v + k1[i] * dt / 2);
  const k2 = deriv(s2);
  const s3 = state.map((v, i) => v + k2[i] * dt / 2);
  const k3 = deriv(s3);
  const s4 = state.map((v, i) => v + k3[i] * dt);
  const k4 = deriv(s4);
  return state.map((v, i) => v + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
}

// ============================================
// Height Simulation (2D vertical plane, no wind)
// State: [x, y, vx, vy]
// ============================================

function simulateHeight(
  profile: Profile,
  env: Environment,
  angle: number,   // bore angle above horizontal (rad)
  targetDist: number
): number {
  const dt = 0.001;
  const rho = airDensity(env);
  const sound = speedOfSound(env.temperature);

  let state = [
    0,
    -profile.scopeHeight / 1000,  // bore starts below scope line
    profile.muzzleVelocity * Math.cos(angle),
    profile.muzzleVelocity * Math.sin(angle),
  ];

  while (state[0] < targetDist) {
    state = rk4Step(state, dt, (s) => {
      const vx = s[2];
      const vy = s[3];
      const v = Math.sqrt(vx * vx + vy * vy);
      if (v < 1) return [vx, vy, 0, -GRAVITY];  // guard: bullet nearly stopped
      const mach = v / sound;
      const drag = dragAcceleration(v, profile.bcG1, mach, rho);
      return [
        vx,
        vy,
        -drag * vx / v,
        -GRAVITY - drag * vy / v,
      ];
    });
  }

  return state[1];
}

// ============================================
// Zero Solver — binary search for bore angle
// that puts bullet on LOS at zeroRange
// ============================================

export function solveZeroAngle(
  profile: Profile,
  env: Environment
): number {
  let low = -0.02;
  let high = 0.02;

  for (let i = 0; i < 30; i++) {
    const mid = (low + high) / 2;
    const y = simulateHeight(profile, env, mid, profile.zeroRange);
    if (y > 0) high = mid;
    else low = mid;
  }

  return (low + high) / 2;
}

// ============================================
// Trajectory Generator
// Lateral drift (wind + spin) computed via lag rule — standard ballistic approximation.
// ============================================

export function generateTrajectory(
  profile: Profile,
  env: Environment,
  wind: Wind
): TrajectoryPoint[] {
  const dt = 0.002;
  const rho = airDensity(env);
  const sound = speedOfSound(env.temperature);
  const massKg = bulletMassKg(profile);

  const zeroAngle = solveZeroAngle(profile, env);
  const windCross = windCrossComponent(wind);

  let state = [
    0,
    -profile.scopeHeight / 1000,
    profile.muzzleVelocity * Math.cos(zeroAngle),
    profile.muzzleVelocity * Math.sin(zeroAngle),
  ];

  let tof = 0;
  let nextSample = TRAJECTORY_STEP;  // FIX: explicit counter, no float modulo
  const table: TrajectoryPoint[] = [];

  while (state[0] < MAX_DISTANCE) {
    state = rk4Step(state, dt, (s) => {
      const vx = s[2];
      const vy = s[3];
      const v = Math.sqrt(vx * vx + vy * vy);
      if (v < 1) return [vx, vy, 0, -GRAVITY];
      const mach = v / sound;
      const drag = dragAcceleration(v, profile.bcG1, mach, rho);
      return [
        vx,
        vy,
        -drag * vx / v,
        -GRAVITY - drag * vy / v,
      ];
    });

    tof += dt;
    const distance = state[0];

    if (distance >= nextSample) {
      const drop = state[1];
      const vx = state[2];
      const vy = state[3];
      const v = Math.sqrt(vx * vx + vy * vy);
      const energy = 0.5 * massKg * v * v;

      if (energy < MIN_ENERGY) break;

      // FIX: lateral drift — lag rule + spin drift
      // Lag rule: drift = windCross * (TOF - range / V0)
      // Physically: bullet "lags" the wind by the difference between actual TOF
      // and the time it would take at muzzle velocity (no drag).
      const lagDrift = windCross * (tof - distance / profile.muzzleVelocity);
      const spinDrift = SPIN_DRIFT_K * tof * tof;  // simplified RH twist model
      const totalDrift = lagDrift + spinDrift;

      // elevationMrad: angle to dial UP to compensate for drop below LOS
      const elevMrad = Math.atan(-drop / distance) * 1000;
      const windMrad = Math.atan(totalDrift / distance) * 1000;

      table.push({
        distance,
        dropCm: drop * 100,
        elevationMrad: elevMrad,
        elevationClicks: clicksFromAngle(elevMrad, profile.clickUnit),
        windDriftCm: totalDrift * 100,
        windMrad,
        windClicks: clicksFromAngle(windMrad, profile.clickUnit),
        velocity: v,
        energy,
      });

      nextSample += TRAJECTORY_STEP;
    }
  }

  return table;
}

// ============================================
// Calculate Shot
// FIX: implemented (was missing entirely)
// Applies shooting angle correction via Rifleman's Rule: cos(angle)
// ============================================

export function calculateShot(
  profile: Profile,
  env: Environment,
  wind: Wind,
  shot: ShotInput
): BallisticResult {
  const traj = generateTrajectory(profile, env, wind);

  if (traj.length === 0) {
    return { elevationMrad: 0, elevationClicks: 0, windageMrad: 0, windageClicks: 0, dropCm: 0, driftCm: 0 };
  }

  // Interpolate trajectory at shot.distance
  let point: TrajectoryPoint = traj[traj.length - 1];

  for (let i = 0; i < traj.length - 1; i++) {
    if (traj[i].distance <= shot.distance && traj[i + 1].distance > shot.distance) {
      const t = (shot.distance - traj[i].distance) / (traj[i + 1].distance - traj[i].distance);
      const a = traj[i];
      const b = traj[i + 1];
      point = {
        distance:        shot.distance,
        dropCm:          a.dropCm          + t * (b.dropCm          - a.dropCm),
        elevationMrad:   a.elevationMrad   + t * (b.elevationMrad   - a.elevationMrad),
        elevationClicks: 0,
        windDriftCm:     a.windDriftCm     + t * (b.windDriftCm     - a.windDriftCm),
        windMrad:        a.windMrad        + t * (b.windMrad        - a.windMrad),
        windClicks:      0,
        velocity:        a.velocity        + t * (b.velocity        - a.velocity),
        energy:          a.energy          + t * (b.energy          - a.energy),
      };
      break;
    }
  }

  // Rifleman's Rule: inclined fire reduces effective gravity component
  const angleRad = (shot.shootingAngle * Math.PI) / 180;
  const elevMrad = point.elevationMrad * Math.cos(angleRad);
  const windMrad = point.windMrad;

  return {
    elevationMrad:   elevMrad,
    elevationClicks: clicksFromAngle(elevMrad, profile.clickUnit),
    windageMrad:     windMrad,
    windageClicks:   clicksFromAngle(windMrad, profile.clickUnit),
    dropCm:          point.dropCm,
    driftCm:         point.windDriftCm,
  };
}

// ============================================
// Click Conversion
// ============================================

export function clicksFromAngle(
  angleMrad: number,
  unit: ClickUnit
): number {
  switch (unit) {
    case "0.1MIL":  return Math.round(angleMrad / 0.1);
    case "0.05MIL": return Math.round(angleMrad / 0.05);
    case "1/4MOA":  return Math.round((angleMrad * 3.438) / 0.25);
    case "1/8MOA":  return Math.round((angleMrad * 3.438) / 0.125);
  }
}
