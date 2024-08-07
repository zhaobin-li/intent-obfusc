# Render all rmds

library(glue)
here::i_am("analysis/render.R")

gen_rmd <- function(iname, type, odir, use_norm) {
  fragment <- TRUE
  if (type == "pdf_document") {
    fragment <- FALSE
  }

  norm <- "None"
  oname <- iname
  imgd <- "imgs"

  if (use_norm) {
    norm <- "0.05"
    oname <- glue("{iname}-normed")
    imgd <- glue("{imgd}-normed")
  }


  xfun::Rscript_call(
    rmarkdown::render,
    list(
      input = glue("{iname}.Rmd"),
      output_format = type, output_file = oname, output_dir = odir,
      params = list(fragment = fragment, norm = norm, img_dir = glue("{imgd}/"))
    )
  )
}

library(here)
setwd(here("analysis"))

write_dir <- "knitted"
xfun::Rscript_call(
  rmarkdown::render,
  list(
    input = "summary.Rmd",
    output_format = "all", output_dir = write_dir,
    params = list(fragment = FALSE)
  )
)

rmds <- c("biased", "randomized", "arbitrary") # arbitrary after randomized
types <- c("pdf_document", "latex_document")
norms <- c(TRUE, FALSE)
for (rn in rmds) {
  for (nm in norms) {
    for (tp in types) {
      gen_rmd(rn, tp, write_dir, use_norm = nm)
    }
    fs::dir_delete(glue("{rn}_cache")) # delete after changing data
  }
}
