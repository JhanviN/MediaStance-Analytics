import { ADV_COLOR, COOP_COLOR, NEUT_COLOR } from "@/lib/constants";

interface Props {
  adv: number;
  coop: number;
  neut: number;
  height?: number;
}

export default function SegmentBar({ adv, coop, neut, height = 5 }: Props) {
  const total = adv + coop + neut || 1;
  return (
    <div
      style={{
        display: "flex",
        height,
        borderRadius: 3,
        overflow: "hidden",
        gap: 1,
      }}
    >
      <div style={{ flex: adv / total, background: ADV_COLOR }} />
      <div style={{ flex: coop / total, background: COOP_COLOR }} />
      <div style={{ flex: neut / total, background: NEUT_COLOR }} />
    </div>
  );
}
