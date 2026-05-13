
function validateScreenResolution(minW, minH) {
    const w = Number(minW) || 0;
    const h = Number(minH) || 0;
    if (w > 0 && window.innerWidth < w) return false;
    if (h > 0 && window.innerHeight < h) return false;
    return true;
}

function getScreenSizes() {
    return {
        "window.screen.height": window.screen.height,
        "document.body.scrollHeight": document.body.scrollHeight,
        "window.innerHeight": window.innerHeight,
        "window.screen.availHeight": window.screen.availHeight,
        "document.body.clientHeight": document.body.clientHeight,
        "window.screen.width": window.screen.width,
        "document.body.scrollWidth": document.body.scrollWidth,
        "window.innerWidth": window.innerWidth,
        "window.screen.availWidth": window.screen.availWidth,
        "document.body.clientWidth": document.body.clientWidth
    };
}