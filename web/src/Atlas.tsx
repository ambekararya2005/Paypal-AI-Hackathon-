/**
 * The attack atlas — criterion 1's screen.
 *
 * 42 cards across six families, and the number that matters is not 42. It is the
 * **implemented** count: cards a registered injector actually generates, checked
 * at import time in both directions by `validate_registry()`. A taxonomy is a
 * document; a taxonomy the generator imports is a dependency. This screen shows
 * both numbers side by side and never lets the larger one stand alone.
 *
 * F5 is empty on purpose and the screen says so where a reader will see it: it
 * is the zero-day holdout family, and an empty family that looks like an
 * oversight is worse than no family at all.
 */

import { useEffect, useMemo, useState } from "react";
import {
  type AtlasCard,
  type AtlasCardDetail,
  type AtlasResponse,
  fetchAtlas,
  fetchAtlasCard,
} from "./api";

const FAMILY_NAMES: Record<string, string> = {
  F1: "Mandate & intent manipulation",
  F2: "Synthetic identity & onboarding",
  F3: "Social engineering at machine speed",
  F4: "Enumeration & probing",
  F5: "Model & memory compromise",
  F6: "Laundering & cash-out",
};

/** The holdout family. Rendered with its reason attached, never as a blank row. */
const HOLDOUT = "F5";

function FamilyBar({ response }: { response: AtlasResponse }) {
  const max = Math.max(...response.families.map((f) => f.total), 1);
  return (
    <div className="panel">
      <div className="panel-head">
        <h2>Six families, 42 cards</h2>
        <span className="hint num">
          {response.implemented} implemented · {response.total - response.implemented} mapped
        </span>
      </div>
      <div className="panel-body">
        <div className="fam-grid">
          {response.families.map((family) => {
            const holdout = family.family === HOLDOUT;
            return (
              <div className="fam" key={family.family}>
                <div className="fam-top">
                  <strong>{family.family}</strong>
                  <span className="hint">{FAMILY_NAMES[family.family] ?? ""}</span>
                </div>
                <div className="fam-track">
                  <div
                    className="fam-fill mapped"
                    style={{ width: `${(family.total / max) * 100}%` }}
                  />
                  <div
                    className="fam-fill impl"
                    style={{ width: `${(family.implemented / max) * 100}%` }}
                  />
                </div>
                <div className="fam-nums num">
                  <span className="impl-n">{family.implemented}</span>
                  <span className="hint"> / {family.total}</span>
                  {holdout && <span className="holdout">zero-day holdout</span>}
                </div>
              </div>
            );
          })}
        </div>
        <div className="note" style={{ marginTop: 14 }}>
          <strong>The implemented count is a ratchet.</strong> A card counts as implemented
          only when an injector exists, is registered, and the generator path on the card
          resolves to a callable in that injector's own module — asserted at package import,
          in both directions, so the atlas and the code cannot disagree. On Day 1 this number
          was 15 on the strength of planned generator paths; making the claim enforceable took
          it down to 8, and it has since moved back up only as code landed.
          <br />
          <br />
          <strong>F5 is empty deliberately.</strong> It is the family the detector never
          trains on, which is what makes the zero-day experiment on the Results screen an
          experiment rather than a demonstration. A test pins the implemented-family set so
          that filling it has to be a decision somebody makes on purpose.
        </div>
      </div>
    </div>
  );
}

function Drawer({ id, onClose }: { id: string; onClose: () => void }) {
  const [card, setCard] = useState<AtlasCardDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    setCard(null);
    setError(null);
    fetchAtlasCard(id)
      .then((c) => live && setCard(c))
      .catch((e) => live && setError(String(e.message ?? e)));
    return () => {
      live = false;
    };
  }, [id]);

  return (
    <div className="drawer-wrap" onClick={onClose}>
      <aside className="drawer" onClick={(e) => e.stopPropagation()}>
        <div className="drawer-head">
          <div>
            <span className="num card-id">{id}</span>
            <h3>{card?.name ?? "..."}</h3>
          </div>
          <button className="btn" onClick={onClose}>
            close
          </button>
        </div>
        {error && <div className="error">{error}</div>}
        {card && (
          <div className="drawer-body">
            <div className="chips">
              <span className={`badge ${card.has_injector ? "impl" : "mapped"}`}>
                {card.has_injector ? "implemented" : "mapped"}
              </span>
              {card.rails.map((r) => (
                <span className="badge rail-b" key={r}>
                  {r}
                </span>
              ))}
              {card.detected_by.map((l) => (
                <span className="badge layer-b" key={l}>
                  {l}
                </span>
              ))}
            </div>

            <p className="desc">{card.description}</p>

            <div className="kv">
              <span>actor</span>
              <span>{card.actor}</span>
            </div>
            <div className="kv">
              <span>GenAI enabler</span>
              <span>{card.genai_enabler}</span>
            </div>
            {card.generator && (
              <div className="kv">
                <span>generator</span>
                <span className="num">{card.generator}</span>
              </div>
            )}

            {/* The load-bearing section. A signal wired to a real feature name is
                what makes a card executable rather than descriptive, so the
                feature column is shown even when it is empty. */}
            <h4>Observable signals</h4>
            <table className="sig">
              <thead>
                <tr>
                  <th>signal</th>
                  <th>feature</th>
                  <th>layer</th>
                </tr>
              </thead>
              <tbody>
                {card.observable_signals.map((s, i) => (
                  <tr key={i}>
                    <td>{s.signal}</td>
                    <td className="num">{s.feature ?? "—"}</td>
                    <td>
                      <span className="badge layer-b">{s.layer}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            {card.preconditions.length > 0 && (
              <>
                <h4>Preconditions</h4>
                <ul>
                  {card.preconditions.map((t, i) => (
                    <li key={i}>{t}</li>
                  ))}
                </ul>
              </>
            )}
            {card.mitigations.length > 0 && (
              <>
                <h4>Mitigations</h4>
                <ul>
                  {card.mitigations.map((t, i) => (
                    <li key={i}>{t}</li>
                  ))}
                </ul>
              </>
            )}
          </div>
        )}
      </aside>
    </div>
  );
}

export default function Atlas() {
  const [data, setData] = useState<AtlasResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [family, setFamily] = useState<string>("all");
  const [onlyImplemented, setOnlyImplemented] = useState(false);
  const [open, setOpen] = useState<string | null>(null);

  useEffect(() => {
    fetchAtlas()
      .then(setData)
      .catch((e) => setError(String(e.message ?? e)));
  }, []);

  const cards = useMemo(() => {
    if (!data) return [];
    return data.cards.filter(
      (c) =>
        (family === "all" || c.family === family) && (!onlyImplemented || c.has_injector),
    );
  }, [data, family, onlyImplemented]);

  if (error) return <div className="error">{error}</div>;
  if (!data) return <div className="loading">loading the atlas…</div>;

  return (
    <>
      <FamilyBar response={data} />

      <div className="panel">
        <div className="panel-head">
          <h2>The cards</h2>
          <div className="controls">
            <select
              className="btn"
              value={family}
              onChange={(e) => setFamily(e.target.value)}
              aria-label="family"
            >
              <option value="all">all families</option>
              {data.families.map((f) => (
                <option key={f.family} value={f.family}>
                  {f.family} — {FAMILY_NAMES[f.family] ?? ""}
                </option>
              ))}
            </select>
            <button
              className={`btn ${onlyImplemented ? "on" : ""}`}
              onClick={() => setOnlyImplemented((v) => !v)}
            >
              implemented only
            </button>
            <span className="hint num">{cards.length} shown</span>
          </div>
        </div>
        <div className="panel-body">
          <div className="card-grid">
            {cards.map((card: AtlasCard) => (
              <button className="atlas-card" key={card.id} onClick={() => setOpen(card.id)}>
                <div className="ac-top">
                  <span className="num card-id">{card.id}</span>
                  <span className={`badge ${card.has_injector ? "impl" : "mapped"}`}>
                    {card.has_injector ? "implemented" : "mapped"}
                  </span>
                </div>
                <div className="ac-name">{card.name}</div>
                <div className="ac-foot">
                  {card.detected_by.slice(0, 3).map((l) => (
                    <span className="badge layer-b" key={l}>
                      {l}
                    </span>
                  ))}
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>

      {data.discovered.length > 0 && (
        <div className="panel">
          <div className="panel-head">
            <h2>Found by the adversarial loop</h2>
            <span className="hint num">{data.discovered.length} variants</span>
          </div>
          <div className="panel-body">
            <div className="card-grid">
              {data.discovered.map((card) => (
                <button className="atlas-card found" key={card.id} onClick={() => setOpen(card.id)}>
                  <div className="ac-top">
                    <span className="num card-id">{card.id}</span>
                    <span className="badge found-b">discovered</span>
                  </div>
                  <div className="ac-name">{card.name}</div>
                </button>
              ))}
            </div>
            <div className="note" style={{ marginTop: 12 }}>
              Variants that survived three or more consecutive rounds against a
              <em> retraining</em> detector, written back with a reproducible genome
              sidecar. They live <strong>beside</strong> the frozen 42 rather than inside it.
              Three further survivors were the <em>unmutated</em> attack — every gene at its
              default — and are deliberately not here: recording those as discoveries would
              claim a find for an attack that was already in the atlas.
            </div>
          </div>
        </div>
      )}

      {open && <Drawer id={open} onClose={() => setOpen(null)} />}
    </>
  );
}
