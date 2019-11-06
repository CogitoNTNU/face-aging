import React, { ChangeEvent } from "react";

import { withStyles } from "@material-ui/core/styles";
import Radio, { RadioProps } from "@material-ui/core/Radio";
import { RadioGroup, FormControlLabel } from "@material-ui/core";

interface Props {
  label: string;
  targetAge: number;
  registerTargetAge: Function;
}

const BlueRadio = withStyles({
  root: {
    "&$checked": {
      color: "#00A7D9"
    }
  },
  checked: {}
})((props: RadioProps) => <Radio color="default" {...props} />);

export default function AgeSelector({
  label,
  registerTargetAge,
  targetAge
}: Props) {
  const handleChange = (event: ChangeEvent<unknown>) => {
    let value = (event.target as HTMLInputElement).value;
    let numericalVal = parseInt(value, 10);
    registerTargetAge(numericalVal);
  };

  return (
    <div>
      <div>{label}</div>
      <RadioGroup
        row
        aria-label="radio-buttons"
        name="ageSelect"
        value={String(targetAge)}
        onChange={handleChange}
      >
        <FormControlLabel
          value={String(0)}
          control={<BlueRadio />}
          label={0}
          labelPlacement="bottom"
        />
        <FormControlLabel
          value={String(1)}
          control={<BlueRadio />}
          label={1}
          labelPlacement="bottom"
        />
        <FormControlLabel
          value={String(2)}
          control={<BlueRadio />}
          label={2}
          labelPlacement="bottom"
        />
        <FormControlLabel
          value={String(3)}
          control={<BlueRadio />}
          label={3}
          labelPlacement="bottom"
        />
        <FormControlLabel
          value={String(4)}
          control={<BlueRadio />}
          label={4}
          labelPlacement="bottom"
        />
        <FormControlLabel
          value={String(5)}
          control={<BlueRadio />}
          label={5}
          labelPlacement="bottom"
        />
      </RadioGroup>
    </div>
  );
}
