
function preference_elicitation() {

}

var updatedOnce = false;
const DEFAULT_VISIBLE_BATCH_SIZE = 20;

function elicitation_ctx_lambda() {
    return {
        "items": Array.from(document.getElementsByTagName("img")).map(x => {
            return {
                "id": x.id, // Corresponds to movie idx
                "name": x.name,
                "url": x.src,
                "title": x.title,
                "viewport": getElementBoundingBox(x)
            };
        }),
    };
}

window.app = new Vue({
    el: '#app',
    delimiters: ['[[',']]'], // Used to replace double { escaping with double [ escaping (to prevent jinja vs vue inference)
    data: function() {
    // let impl = document.getElementById("impl").className;
    return {
        userEmail: "{{email}}",
        items: [],
        selected: [],
        impl: impl,
        selectMode: "multi",
        lastSelectedCluster: null,
        handle: false,
        rows: [],
        rows2: [],
        itemsPerRow: 80,
        jumboHeader: "Preference Elicitation",
        disableNextStep: false,
        searchMovieName: null,
        itemsBackup: null,
        displayedCountBackup: null,
        rowsBackup: null,
        busy: false,
        visibleBatchSize: 0,
        loadingMore: false,
        displayedCount: 0
    }
    },
    computed: {
        visibleItems() {
            return Array.isArray(this.items) ? this.items.slice(0, this.displayedCount || this.items.length) : [];
        },
        hasActiveSearch() {
            return this.itemsBackup !== null;
        }
    },
    async mounted() {
        const btns = document.querySelectorAll(".btn");
        
        // This was used for reporting as previously reporting endpoints were defined inside plugin

        // Get the number of items user is supposed to select
        await this.ensureVisibleItems(DEFAULT_VISIBLE_BATCH_SIZE, true);
        this.visibleBatchSize = Math.min(DEFAULT_VISIBLE_BATCH_SIZE, this.items.length);

        // Register the handlers for event reporting
        startViewportChangeReportingWithLimit(`/utils/changed-viewport`, csrfToken, 1.0, true, elicitation_ctx_lambda);
        registerClickedButtonReporting(`/utils/on-input`, csrfToken, btns, ()=>{
            return {
                "search_text_box_value": this.searchMovieName
            };
        });
        reportLoadedPage(`/utils/loaded-page`, csrfToken, "preference_elicitation", ()=>{
            return {"impl":impl};
        });
        
        setTimeout(function () {
            reportViewportChange(`/utils/changed-viewport`, csrfToken, elicitation_ctx_lambda);
        }, 5000);
    },
    methods: {
        parseMovieMeta(dat) {
            const rawTitle = dat["movie"] || "";
            const genres = Array.isArray(dat["genres"])
                ? dat["genres"].filter((genre) => genre && genre !== "(no genres listed)")
                : [];

            let displayTitle = rawTitle.replace(/\s*\(no genres listed\)\s*$/i, "").trim();
            if (genres.length > 0) {
                const yearMatch = displayTitle.match(/^(.*\(\d{4}\))/);
                if (yearMatch) {
                    displayTitle = yearMatch[1];
                }
                const suffix = " " + genres.join("|");
                if (displayTitle.endsWith(suffix)) {
                    displayTitle = displayTitle.slice(0, displayTitle.length - suffix.length);
                }
            }

            return {
                "displayTitle": displayTitle.trim(),
                "genreList": genres
            };
        },
        async handlePrefixSearch(movieName) {
            let foundMovies = await fetch(search_item_url + "?attrib=movie&pattern="+movieName).then(resp => resp.json());
            return foundMovies;
        },
        makeItem(dat) {
            const meta = this.parseMovieMeta(dat);
            return {
                "movieName": dat["movie"],
                "displayTitle": meta["displayTitle"] || dat["movie"],
                "genreList": meta["genreList"],
                "movie": {
                    "idx": dat["movie_idx"],
                    "url": dat["url"]
                }
            };
        },
        hasPoster(dat) {
            return !!(dat && dat["url"] && String(dat["url"]).trim().length > 0);
        },
        async fetchElicitationData(isInitial=false) {
            const suffix = isInitial ? "&i=0" : "";
            return await fetch(initial_data_url + "?impl=" + this.impl + suffix).then((resp) => resp.json()).then((resp) => resp);
        },
        prepareTable(data, fromSearch=false) {
            let row = [];
            let rows = [];
            let items = [];
            for (var k in data) {
                if (!this.hasPoster(data[k])) {
                    continue;
                }
                let it = this.makeItem(data[k]);
                let rw = this.makeItem(data[k]);
                
                if (fromSearch === true) {
                    it["_fromSearch"] = true;
                    rw["_fromSearch"] = true;
                }
                
                items.push(it);
                row.push(rw);
                if (row.length >= this.itemsPerRow) {
                    rows.push(row);
                    row = [];
                }
            }
            if (row.length > 0) {
                rows.push(row);
            }

            return {"rows": rows, "items": items };
        },
        async ensureVisibleItems(targetCount, isInitial=false) {
            let res = {"rows": this.rows, "items": this.items};
            let previousVisibleCount = -1;

            for (let attempt = 0; attempt < 8; attempt++) {
                const data = await this.fetchElicitationData(isInitial && attempt === 0);
                res = this.prepareTable(data);
                this.rows = res["rows"];
                this.items = res["items"];
                this.displayedCount = Math.min(targetCount, this.items.length);

                if (this.items.length >= targetCount) {
                    break;
                }
                if (this.items.length === previousVisibleCount) {
                    break;
                }
                previousVisibleCount = this.items.length;
            }

            return res;
        },
        async onClickSearch(event) {
            let data = await this.handlePrefixSearch(this.searchMovieName);
            let res = this.prepareTable(data, true);
            reportOnInput("/utils/on-input", csrfToken, "search", {"search_text_box_value": this.searchMovieName, "search_result": res});

            // Do not overwrite backups when doing repeated search
            if (this.itemsBackup === null) {
                this.itemsBackup = this.items;
                this.displayedCountBackup = this.displayedCount;
                this.rowsBackup = this.rows;
            }

            this.rows = res["rows"];
            this.items = res["items"];
            this.displayedCount = this.items.length;
        },
        onKeyDownSearchMovieName(e) {
            if (e.key === "Enter") {
                this.onClickSearch(null);
            }
        },
        onClickCancelSearch() {
            this.items = this.itemsBackup;
            this.displayedCount = this.displayedCountBackup;
            this.rows = this.rowsBackup;
            this.itemsBackup = null;
            this.displayedCountBackup = null;
            this.rowsBackup = null;
        },
        async onClickLoadMore() {
            const previousCount = this.displayedCount || this.items.length;
            const targetIncrease = this.visibleBatchSize || previousCount || 1;
            const targetCount = previousCount + targetIncrease;

            this.loadingMore = true;
            try {
                await this.ensureVisibleItems(targetCount, false);
            } finally {
                this.loadingMore = false;
            }
        },
        onUpdateSearchMovieName(newValue) {
        },
        isMovieSelected(item) {
            return this.movieIndexOf(this.selected, item) > -1;
        },
        movieIndexOf(arr, item) {
            for (let idx in arr) {
                let arrItem = arr[idx];
                if (arrItem.movie.idx === item.movie.idx
                    && arrItem.movieName === item.movieName
                    && arrItem.movie.url === item.movie.url) {
                        return idx;
                    }
            }
            return -1;
        },
        onSelectMovie(event, item) {
            // TODO wrap movieIndexOf as generic indexOf with selector lambda
            let index = this.movieIndexOf(this.selected, item); //this.selected.indexOf(item);
            if (index > -1) {
                // Already there, remove it
                this.selected.splice(index, 1);
                reportDeselectedItem(`/utils/deselected-item`, csrfToken, item, this.selected);
            } else {
                // Not there, insert
                this.selected.push(item);
                reportSelectedItem(`/utils/selected-item`, csrfToken, item, this.selected);
            }
        },
        onRowClicked(item) {
            let index = this.movieIndexOf(this.selected, item); // this.selected.indexOf(item);
            if (index > -1) {
                this.selected.splice(index, 1);
                //this.$refs.selectableTable.unselectRow(this.items.indexOf(item));
            } else {
                this.selected.push(item);
                //this.$refs.selectableTable.selectRow(this.items.indexOf(item));
            }
        },
        onElicitationFinish(form) {
            if (this.selected.length < 5) {
                this.$bvModal.show('bv-modal-example');
                return;
            }
            this.busy = true;
            let selectedMoviesTag = document.createElement("input");
            selectedMoviesTag.setAttribute("type","hidden");
            selectedMoviesTag.setAttribute("name","selectedMovies");
            selectedMoviesTag.setAttribute("value", this.selected.map((x) => x.movie.idx).join(","));

            form.appendChild(selectedMoviesTag);

            form.submit();
        },
        itemMouseEnter(event) {
            reportOnInput("/utils/on-input", csrfToken, "mouse-enter", {
                "target": {
                    "id": event.target.id,
                    "name": event.target.name,
                    "alt": event.target.alt,
                    "title": event.target.title
                }
            });
        },
        itemMouseLeave(event) {
            reportOnInput("/utils/on-input", csrfToken, "mouse-leave", {
                "target": {
                    "id": event.target.id,
                    "name": event.target.name,
                    "alt": event.target.alt,
                    "title": event.target.title
                }
            });
        },
    }
})