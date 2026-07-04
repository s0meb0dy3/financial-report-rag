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
    | "pencil"
    | "plus"
    | "refresh"
    | "search"
    | "stop"
    | "trash"
    | "upload";
  className?: string;
}) {
  const paths = {
    bolt: "M13 2 4 14h7l-1 8 10-13h-7l1-7Z",
    chevron: "m6 9 6 6 6-6",
    copy: "M8 8h10a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2V10a2 2 0 0 1 2-2Zm-2 8H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1",
    file: "M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Zm0 0v6h6M8 13h8M8 17h5",
    menu: "M7 4h10M7 12h10M7 20h10",
    panel: "M4 5h16v14H4zM14 5v14",
    pencil: "m4 20 4.5-1 10-10a2.1 2.1 0 0 0-3-3l-10 10L4 20Zm11-13 2 2",
    plus: "M12 5v14M5 12h14",
    refresh: "M20 6v5h-5M4 18v-5h5M18.5 10A6.5 6.5 0 0 0 7 7.2L4 11m16 2-3 3.8A6.5 6.5 0 0 1 5.5 14",
    search: "m21 21-4.3-4.3M10.5 18a7.5 7.5 0 1 1 0-15 7.5 7.5 0 0 1 0 15Z",
    stop: "M8 8h8v8H8z",
    trash: "M4 7h16M10 11v6M14 11v6M6 7l1 14h10l1-14M9 7V4h6v3",
    upload: "M12 16V4m0 0 5 5m-5-5-5 5M5 20h14",
  };
  return (
    <svg className={`icon ${className}`} viewBox="0 0 24 24" aria-hidden="true">
      <path d={paths[name]} />
    </svg>
  );
}
