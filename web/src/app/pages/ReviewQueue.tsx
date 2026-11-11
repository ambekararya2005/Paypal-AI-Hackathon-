/** Review queue: held orders waiting on a person. Funds stay reserved until they decide. */

import { TimerIcon } from "@radix-ui/react-icons";
import { consentHref, layerWords, merchantName, usd, useGateLog } from "../data";
import { orderHref } from "../router";
import { Empty, PageHead, Panel, Skeleton, TableFoot, When } from "../ui";

export default function ReviewQueue() {
  const log = useGateLog();
  const held = log.rows.filter((r) => r.final_state === "held");
  return (
    <>
      <PageHead
        title="Review queue"
        sub="Orders MANTIS held for a person. The funds are reserved on PayPal; approving captures them, denying voids the authorization."
      />
      <Panel flush>
        {log.status === "loading" ? (
          <div className="m-pad"><Skeleton rows={3} /></div>
        ) : log.status === "error" ? (
          <Empty title="The gate log is not reachable">This copy of the site has no API.</Empty>
        ) : held.length === 0 ? (
          <Empty icon={<TimerIcon />} title="Nothing waiting.">
            Held orders appear here when MANTIS isn’t sure.
          </Empty>
        ) : (
          <>
            <div className="m-table-wrap">
              <table className="m-table">
                <thead>
                  <tr>
                    <th>Time</th>
                    <th>Merchant</th>
                    <th className="m-r">Amount</th>
                    <th>Reason</th>
                    <th className="m-hide-sm">Order ID</th>
                    <th aria-label="Action" />
                  </tr>
                </thead>
                <tbody>
                  {held.map((r) => (
                    <tr key={r.order_id}>
                      <td className="m-nowrap m-muted"><When ts={r.ts} /></td>
                      <td>{merchantName(r.merchant_id)}</td>
                      <td className="m-r num">{usd(r.total, r.currency)}</td>
                      <td>{layerWords(r.layers_fired)}</td>
                      <td className="m-hide-sm">
                        <a className="mono" href={orderHref(r.order_id)}>{r.order_id}</a>
                      </td>
                      <td className="m-r">
                        <a className="m-btn m-btn-sm m-btn-primary" href={consentHref(r.order_id)}>Review</a>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <TableFoot shown={held.length} total={held.length} />
          </>
        )}
      </Panel>
    </>
  );
}
