// Keep table labels visible without changing Material's horizontal scrolling.
function installTableHeaders() {
  const cleanups = [];
  document.querySelectorAll('.md-typeset table').forEach((table) => {
    if (!table.tHead) return;
    const scroller = table.closest('.md-typeset__scrollwrap') || table.parentElement;
    const overlay = document.createElement('div');
    overlay.className = 'sticky-table-header';
    overlay.hidden = true;
    overlay.setAttribute('aria-hidden', 'true');
    overlay.inert = true;
    const copy = table.cloneNode(false);
    copy.removeAttribute('id');
    copy.append(table.tHead.cloneNode(true));
    copy.querySelectorAll('[id]').forEach((node) => node.removeAttribute('id'));
    overlay.append(copy);
    document.body.append(overlay);

    const update = () => {
      const rect = table.getBoundingClientRect();
      const head = table.tHead.getBoundingClientRect();
      const clip = scroller.getBoundingClientRect();
      const header = document.querySelector('.md-header');
      const top = Math.max(0, header?.getBoundingClientRect().bottom || 0);
      const left = Math.max(0, rect.left, clip.left);
      const right = Math.min(window.innerWidth, rect.right, clip.right);
      overlay.hidden = head.top >= top || rect.bottom <= top || right <= left;
      if (overlay.hidden) return;
      Object.assign(overlay.style, {
        top: `${Math.min(top, rect.bottom - head.height)}px`,
        left: `${left}px`,
        width: `${right - left}px`,
      });
      copy.style.width = `${rect.width}px`;
      copy.style.marginLeft = `${rect.left - left}px`;
      const originals = table.tHead.querySelectorAll('th, td');
      copy.querySelectorAll('th, td').forEach((cell, index) => {
        const source = originals[index];
        const style = getComputedStyle(source);
        // Computed font shorthand can be empty; copy typography explicitly.
        for (const property of [
          'padding', 'font-family', 'font-size', 'font-weight', 'font-style',
          'font-stretch', 'font-variant', 'line-height', 'letter-spacing',
          'word-spacing', 'text-transform', 'text-align', 'border', 'box-sizing',
        ]) {
          cell.style.setProperty(property, style.getPropertyValue(property));
        }
        cell.style.width = `${source.getBoundingClientRect().width}px`;
      });
    };
    window.addEventListener('scroll', update, true);
    window.addEventListener('resize', update);
    const observer = new ResizeObserver(update);
    observer.observe(table);
    update();
    cleanups.push(() => {
      window.removeEventListener('scroll', update, true);
      window.removeEventListener('resize', update);
      observer.disconnect();
      overlay.remove();
    });
  });
  return () => cleanups.forEach((cleanup) => cleanup());
}

let removeTableHeaders = () => {};
// Material emits document$ after initial load and each instant-navigation visit.
document$.subscribe(() => {
  removeTableHeaders();
  removeTableHeaders = installTableHeaders();
});
