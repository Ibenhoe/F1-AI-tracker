// src/components/racereplay/utils/telemetry.js

export function toPct(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return 0;
  if (n >= 0 && n <= 1) return n * 100; // 0..1 -> %
  return n; // al in %
}

export function toBoolDRS(v) {
  if (v === true) return true;
  if (v === false) return false;
  const n = Number(v);
  if (Number.isFinite(n)) return n > 0;
  const s = String(v ?? "").toLowerCase();
  return s === "on" || s === "true" || s === "active" || s === "1" || s === "yes";
}

// Heuristiek: als speed klein is, is het waarschijnlijk m/s -> km/h
export function toKmh(speed) {
  const n = Number(speed);
  if (!Number.isFinite(n)) return 0;
  return n < 60 ? n * 3.6 : n;
}

export function normalizeDriver(driverData) {
  const position = Number.isFinite(Number(driverData?.position)) ? Math.round(Number(driverData.position)) : 0;
  const speedKmh = toKmh(driverData?.speed);
  const throttlePct = toPct(driverData?.throttle);
  const brakePct = toPct(driverData?.brake);
  const drsOn = toBoolDRS(driverData?.drs);

  return {
    position,
    speedKmh,
    gear: driverData?.gear ?? "—",
    throttlePct,
    brakePct,
    tireCompound: driverData?.tire_compound ?? "—",
    tireAge: Number.isFinite(Number(driverData?.tire_age)) ? Number(driverData.tire_age) : 0,
    drsOn,
    gap: driverData?.gap ?? "—",
    pitStops: driverData?.pit_stops ?? 0,
    lapTime: driverData?.lap_time ?? null,
    status: driverData?.status ?? null,
    driverName: driverData?.driver_name ?? null,
  };
}