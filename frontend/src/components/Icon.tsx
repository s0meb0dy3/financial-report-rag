export function Icon({
  name,
  className = "",
}: {
  name:
    | "bolt"
    | "chevron"
    | "copy"
    | "file"
    | "menu"
    | "panel"
    | "plus";
  className?: string;
}) {
  const paths = {
    bolt: "M13 2 4 14h7l-1 8 10-13h-7l1-7Z",
    chevron: "m6 9 6 6 6-6",
    copy: "M8 8h10a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2V10a2 2 0 0 1 2-2Zm-2 8H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1",
    file: "M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Zm0 0v6h6M8 13h8M8 17h5",
    menu: "M7 4h10M7 12h10M7 20h10",
    panel: "M4 5h16v14H4zM14 5v14",
    plus: "M12 5v14M5 12h14",
  };
  return (
    <svg className={`icon ${className}`} viewBox="0 0 24 24" aria-hidden="true">
      <path d={paths[name]} />
    </svg>
  );
}
