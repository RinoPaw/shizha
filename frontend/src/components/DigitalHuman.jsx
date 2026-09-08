/**
 * The officer is intentionally a stable visual anchor. Speech state changes
 * the subtle frame treatment only; there are no video clips to swap while a
 * voice turn is starting, speaking, or being interrupted.
 */
export function DigitalHuman({ mode = "idle" }) {
  const normalizedMode = mode === "speaking" ? "speaking" : "idle";

  return (
    <div className={`human-stage human-stage-${normalizedMode}`}>
      <img
        className="human-poster"
        src="/assets/portraits/panda-officer.jpg"
        alt="熊猫警官"
      />
      <div className="human-vignette" />
      <div className="human-caption">
        <div className="human-brand">
          <div>
            <strong>熊猫警官</strong>
            <small>识诈安全助手</small>
          </div>
        </div>
      </div>
    </div>
  );
}

export default DigitalHuman;
